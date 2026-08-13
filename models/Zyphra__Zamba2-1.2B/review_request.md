# 검토 의뢰서 — Zyphra/Zamba2-1.2B

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `zamba2`
- 판단 필요: **6건**

## 증거 — 이미 받아둔 실제 소스

- `develop/sources/modeling_zamba2.py` — 있음, 이 파일을 열어서 판정한다
- `develop/sources/configuration_zamba2.py` — 있음, 이 파일을 열어서 판정한다

- 온라인 원본: https://github.com/huggingface/transformers/tree/main/src/transformers/models/zamba2

그 밖의 재료: `full/review.md`(리뷰 패킷 — shape 별 실제 행 표본), `structure.yaml`(이 모델의 심볼 표), `full/<phase>.csv`(전체 operator 표).

## 판단이 필요한 것

### 6. 값이 겹쳐 **임의로** 고른 축

두 심볼이 같은 값을 갖는 자리다. 규칙에는 고를 근거가 없고, 이긴 쪽은 전역 우선순위 — 즉 **관례**로 정해졌다. 이름이 맞을 수도 있지만 파이프라인은 그걸 알지 못한다. 표에서는 확신 있는 라벨과 똑같이 보인다.

**소스를 열어 어느 쪽인지 확정하는 것이 여기서 할 일이다.** 확정되면 `rules/label_overrides.yaml` 에 근거와 함께 못 박는다(review/05-overrides.md). 출신으로만 구별되는 경우라면 그렇게 적고 `open` 으로 남긴다.

- `d_head_ssm vs n_h_ssm` in `model.layers.*.mamba` — 값 64 를 두고 후보가 2개, 9440축
- `d_head_ssm vs n_h_ssm` in `model.layers.*.mamba_decoder.mamba` — 값 64 를 두고 후보가 2개, 1770축
- `n_h vs n_kv` in `model.layers.*.shared_transformer.self_attn` — 값 32 를 두고 후보가 2개, 1020축
- `d_head vs r_lora` in `model.layers.*.shared_transformer.self_attn.linear_q_adapter_list.*.*` — 값 128 를 두고 후보가 2개, 194축
- `d_head vs r_lora` in `model.layers.*.shared_transformer.self_attn.linear_k_adapter_list.*.*` — 값 128 를 두고 후보가 2개, 194축
- `d_head vs r_lora` in `model.layers.*.shared_transformer.self_attn.linear_v_adapter_list.*.*` — 값 128 를 두고 후보가 2개, 194축

## 기계적으로 이미 확인된 것 — 다시 묻지 말 것

- **심볼이 읽은 config 필드**: 전부 이 모델의 config 클래스(또는 상속/프로퍼티/getattr 기본값)에 존재한다
- **정사각 축**: `d_chunk` ← 소스의 `chunk_size` ← `chunk_size`
- **모듈이 읽는 config 속성**: `__init__` 에서 config 를 읽는 클래스 9개를 소스에서 확인했다. 그 목록이 각 모듈의 폭이 가질 수 있는 이름의 전부다.
- **가중치 축 ↔ 모듈 소속**: 가중치 축의 이름이 전부 그 모듈(또는 그 부모)이 실제로 읽는 config 필드에서 나왔다. 이 축들은 값이 아니라 소스로 확인된 것이다.

## 행 단위 전건 — 여기부터 읽는다

접힌 표(`<phase>.jsonl`)의 **고유 행 전부**다. 레이어 인덱스만 접었고 그 밖에는 아무것도 합치지 않았다. 아래 A/B/C 절은 (모듈, 라벨)로 접은 뷰라 **한 행 안의 어긋남이 보이지 않는다** — 실제로 외부 검토가 찾아낸 결함 세 건이 전부 그 자리에 있었다.

**한 행씩 읽고 이것만 물어라: 입력·가중치·출력이 서로 말이 되는가.**

- 같은 텐서가 `input_shape` 와 `weight_shape` 에서 다른 이름을 쓰지 않는가 (가중치는 `[out, in]` 으로 저장되고 피연산자는 전치돼 있다)
- 전치·view 처럼 **이름을 바꿀 수 없는 op** 이 이름을 바꾸지 않았는가
- 행렬곱의 수축 축이 양쪽에서 같은 이름인가 — `[m,k] @ [k,n] -> [m,n]`
- 이 모듈이 그 이름을 가질 수 있는가 (소스에서 그 `nn.Linear` 를 만드는 줄을 찾아라)

고유 행 94개.

| phase | 모듈 | op | input_shape | weight_shape | output_shape |
|---|---|---|---|---|---|
| prefill | `model.embed_tokens` | embedding | `[['V', 'd_model'], ['B', 'T']]` | `['V', 'd_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.input_layernorm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.mamba.in_proj` | matmul | `[['T', 'd_model'], ['d_model', '2*d_inner+2*n_g*d_state+n_h_ssm']]` | `['2*d_inner+2*n_g*d_state+n_h_ssm', 'd_model']` | `[['T', '2*d_inner+2*n_g*d_state+n_h_ssm']]` |
| prefill | `model.layers.*.mamba.conv1d` | conv1d | `[['B', 'd_inner+2*n_g*d_state', 'T'], ['d_inner+2*n_g*d_state', '1', 'd_conv'], ['d_inner+2*n_g*d_state']]` | `['d_inner+2*n_g*d_state', '1', 'd_conv']` | `[['B', 'd_inner+2*n_g*d_state', 'T+d_conv-1']]` |
| prefill | `model.layers.*.mamba.act` | silu | `[['B', 'T', 'd_inner+2*n_g*d_state']]` | `None` | `[['B', 'T', 'd_inner+2*n_g*d_state']]` |
| prefill | `model.layers.*.mamba` | exp | `[['d_head_ssm']]` | `None` | `[['d_head_ssm']]` |
| prefill | `model.layers.*.mamba` | exp | `[['B', 'd_head_ssm', '1', 'd_chunk', 'd_chunk']]` | `None` | `[['B', 'd_head_ssm', '1', 'd_chunk', 'd_chunk']]` |
| prefill | `model.layers.*.mamba` | exp | `[['B', 'd_head_ssm', '1', 'd_chunk']]` | `None` | `[['B', 'd_head_ssm', '1', 'd_chunk']]` |
| prefill | `model.layers.*.mamba` | exp | `[['B', 'd_head_ssm', '2', '2']]` | `None` | `[['B', 'd_head_ssm', '2', '2']]` |
| prefill | `model.layers.*.mamba.norm` | rmsnorm | `[['B', 'T', 'd_inner']]` | `['d_inner']` | `[['B', 'T', 'd_inner']]` |
| prefill | `model.layers.*.mamba.out_proj` | matmul | `[['T', 'd_inner'], ['d_inner', 'd_model']]` | `['d_model', 'd_inner']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_add | `[['B', 'T', 'd_model'], ['B', 'T', 'd_model']]` | `None` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.shared_transformer` | concat | `[['B', 'T', 'd_model'], ['B', 'T', 'd_model']]` | `None` | `[['B', 'T', 'd_attn']]` |
| prefill | `model.layers.*.shared_transformer.input_layernorm` | rmsnorm | `[['B', 'T', 'd_attn']]` | `['d_attn']` | `[['B', 'T', 'd_attn']]` |
| prefill | `model.layers.*.shared_transformer.self_attn.q_proj` | matmul | `[['T', 'd_attn'], ['d_attn', 'd_attn']]` | `['d_attn', 'd_attn']` | `[['T', 'd_attn']]` |
| prefill | `model.layers.*.shared_transformer.self_attn.k_proj` | matmul | `[['T', 'd_attn'], ['d_attn', 'd_attn']]` | `['d_attn', 'd_attn']` | `[['T', 'd_attn']]` |
| prefill | `model.layers.*.shared_transformer.self_attn.v_proj` | matmul | `[['T', 'd_attn'], ['d_attn', 'd_attn']]` | `['d_attn', 'd_attn']` | `[['T', 'd_attn']]` |
| prefill | `model.layers.*.shared_transformer.self_attn.linear_q_adapter_list.*.*` | matmul | `[['T', 'd_attn'], ['d_attn', 'r_lora']]` | `['r_lora', 'd_attn']` | `[['T', 'r_lora']]` |
| prefill | `model.layers.*.shared_transformer.self_attn.linear_q_adapter_list.*.*` | matmul | `[['T', 'r_lora'], ['r_lora', 'd_attn']]` | `['d_attn', 'r_lora']` | `[['T', 'd_attn']]` |
| prefill | `model.layers.*.shared_transformer.self_attn.linear_k_adapter_list.*.*` | matmul | `[['T', 'd_attn'], ['d_attn', 'r_lora']]` | `['r_lora', 'd_attn']` | `[['T', 'r_lora']]` |
| prefill | `model.layers.*.shared_transformer.self_attn.linear_k_adapter_list.*.*` | matmul | `[['T', 'r_lora'], ['r_lora', 'd_attn']]` | `['d_attn', 'r_lora']` | `[['T', 'd_attn']]` |
| prefill | `model.layers.*.shared_transformer.self_attn.linear_v_adapter_list.*.*` | matmul | `[['T', 'd_attn'], ['d_attn', 'r_lora']]` | `['r_lora', 'd_attn']` | `[['T', 'r_lora']]` |
| prefill | `model.layers.*.shared_transformer.self_attn.linear_v_adapter_list.*.*` | matmul | `[['T', 'r_lora'], ['r_lora', 'd_attn']]` | `['d_attn', 'r_lora']` | `[['T', 'd_attn']]` |
| prefill | `model.layers.*.shared_transformer.self_attn` | batched_matmul | `[['n_h', 'T', 'd_head'], ['n_h', 'd_head', 'T']]` | `None` | `[['n_h', 'T', 'T']]` |
| prefill | `model.layers.*.shared_transformer.self_attn` | softmax | `[['B', 'n_h', 'T', 'T']]` | `None` | `[['B', 'n_h', 'T', 'T']]` |
| prefill | `model.layers.*.shared_transformer.self_attn` | batched_matmul | `[['n_h', 'T', 'T'], ['n_h', 'T', 'd_head']]` | `None` | `[['n_h', 'T', 'd_head']]` |
| prefill | `model.layers.*.shared_transformer.self_attn.o_proj` | matmul | `[['T', 'd_attn'], ['d_attn', 'd_model']]` | `['d_model', 'd_attn']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*.shared_transformer.pre_ff_layernorm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.shared_transformer.feed_forward.gate_up_proj` | matmul | `[['T', 'd_model'], ['d_model', '2*d_ff']]` | `['2*d_ff', 'd_model']` | `[['T', '2*d_ff']]` |
| prefill | `model.layers.*.shared_transformer.feed_forward.gate_up_proj_adapter_list.*.*` | matmul | `[['T', 'd_model'], ['d_model', 'r_lora']]` | `['r_lora', 'd_model']` | `[['T', 'r_lora']]` |
| prefill | `model.layers.*.shared_transformer.feed_forward.gate_up_proj_adapter_list.*.*` | matmul | `[['T', 'r_lora'], ['r_lora', '2*d_ff']]` | `['2*d_ff', 'r_lora']` | `[['T', '2*d_ff']]` |
| prefill | `model.layers.*.shared_transformer.feed_forward` | elementwise_add | `[['B', 'T', '2*d_ff'], ['B', 'T', '2*d_ff']]` | `None` | `[['B', 'T', '2*d_ff']]` |
| prefill | `model.layers.*.shared_transformer.feed_forward.act_fn` | gelu | `[['B', 'T', 'd_ff']]` | `None` | `[['B', 'T', 'd_ff']]` |
| prefill | `model.layers.*.shared_transformer.feed_forward` | elementwise_mul | `[['B', 'T', 'd_ff'], ['B', 'T', 'd_ff']]` | `None` | `[['B', 'T', 'd_ff']]` |
| prefill | `model.layers.*.shared_transformer.feed_forward.down_proj` | matmul | `[['T', 'd_ff'], ['d_ff', 'd_model']]` | `['d_model', 'd_ff']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*.linear` | matmul | `[['T', 'd_model'], ['d_model', 'd_model']]` | `['d_model', 'd_model']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*.mamba_decoder` | elementwise_add | `[['B', 'T', 'd_model'], ['B', 'T', 'd_model']]` | `None` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.mamba_decoder.input_layernorm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.mamba_decoder.mamba.in_proj` | matmul | `[['T', 'd_model'], ['d_model', '2*d_inner+2*n_g*d_state+n_h_ssm']]` | `['2*d_inner+2*n_g*d_state+n_h_ssm', 'd_model']` | `[['T', '2*d_inner+2*n_g*d_state+n_h_ssm']]` |
| prefill | `model.layers.*.mamba_decoder.mamba.conv1d` | conv1d | `[['B', 'd_inner+2*n_g*d_state', 'T'], ['d_inner+2*n_g*d_state', '1', 'd_conv'], ['d_inner+2*n_g*d_state']]` | `['d_inner+2*n_g*d_state', '1', 'd_conv']` | `[['B', 'd_inner+2*n_g*d_state', 'T+d_conv-1']]` |
| prefill | `model.layers.*.mamba_decoder.mamba.act` | silu | `[['B', 'T', 'd_inner+2*n_g*d_state']]` | `None` | `[['B', 'T', 'd_inner+2*n_g*d_state']]` |
| prefill | `model.layers.*.mamba_decoder.mamba` | exp | `[['d_head_ssm']]` | `None` | `[['d_head_ssm']]` |
| prefill | `model.layers.*.mamba_decoder.mamba` | exp | `[['B', 'd_head_ssm', '1', 'd_chunk', 'd_chunk']]` | `None` | `[['B', 'd_head_ssm', '1', 'd_chunk', 'd_chunk']]` |
| prefill | `model.layers.*.mamba_decoder.mamba` | exp | `[['B', 'd_head_ssm', '1', 'd_chunk']]` | `None` | `[['B', 'd_head_ssm', '1', 'd_chunk']]` |
| prefill | `model.layers.*.mamba_decoder.mamba` | exp | `[['B', 'd_head_ssm', '2', '2']]` | `None` | `[['B', 'd_head_ssm', '2', '2']]` |
| prefill | `model.layers.*.mamba_decoder.mamba.norm` | rmsnorm | `[['B', 'T', 'd_inner']]` | `['d_inner']` | `[['B', 'T', 'd_inner']]` |
| prefill | `model.layers.*.mamba_decoder.mamba.out_proj` | matmul | `[['T', 'd_inner'], ['d_inner', 'd_model']]` | `['d_model', 'd_inner']` | `[['T', 'd_model']]` |
| prefill | `model.final_layernorm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `lm_head` | matmul | `[['T', 'd_model'], ['d_model', 'V']]` | `['V', 'd_model']` | `[['T', 'V']]` |
| decode | `model.embed_tokens` | embedding | `[['V', 'd_model'], ['B', '1']]` | `['V', 'd_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.input_layernorm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.mamba.in_proj` | matmul | `[['B', 'd_model'], ['d_model', '2*d_inner+2*n_g*d_state+n_h_ssm']]` | `['2*d_inner+2*n_g*d_state+n_h_ssm', 'd_model']` | `[['B', '2*d_inner+2*n_g*d_state+n_h_ssm']]` |
| decode | `model.layers.*.mamba.act` | silu | `[['B', 'd_inner+2*n_g*d_state']]` | `None` | `[['B', 'd_inner+2*n_g*d_state']]` |
| decode | `model.layers.*.mamba` | exp | `[['d_head_ssm']]` | `None` | `[['d_head_ssm']]` |
| decode | `model.layers.*.mamba` | exp | `[['B', 'd_head_ssm', 'n_h_ssm', 'd_state']]` | `None` | `[['B', 'd_head_ssm', 'n_h_ssm', 'd_state']]` |
| decode | `model.layers.*.mamba` | batched_matmul | `[['d_head_ssm', 'n_h_ssm', 'd_state'], ['d_head_ssm', 'd_state', 'B']]` | `None` | `[['d_head_ssm', 'n_h_ssm', 'B']]` |
| decode | `model.layers.*.mamba.norm` | rmsnorm | `[['B', '1', 'd_inner']]` | `['d_inner']` | `[['B', '1', 'd_inner']]` |
| decode | `model.layers.*.mamba.out_proj` | matmul | `[['B', 'd_inner'], ['d_inner', 'd_model']]` | `['d_model', 'd_inner']` | `[['B', 'd_model']]` |
| decode | `model.layers.*` | elementwise_add | `[['B', '1', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.shared_transformer` | concat | `[['B', '1', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '1', 'd_attn']]` |
| decode | `model.layers.*.shared_transformer.input_layernorm` | rmsnorm | `[['B', '1', 'd_attn']]` | `['d_attn']` | `[['B', '1', 'd_attn']]` |
| decode | `model.layers.*.shared_transformer.self_attn.q_proj` | matmul | `[['B', 'd_attn'], ['d_attn', 'd_attn']]` | `['d_attn', 'd_attn']` | `[['B', 'd_attn']]` |
| decode | `model.layers.*.shared_transformer.self_attn.k_proj` | matmul | `[['B', 'd_attn'], ['d_attn', 'd_attn']]` | `['d_attn', 'd_attn']` | `[['B', 'd_attn']]` |
| decode | `model.layers.*.shared_transformer.self_attn.v_proj` | matmul | `[['B', 'd_attn'], ['d_attn', 'd_attn']]` | `['d_attn', 'd_attn']` | `[['B', 'd_attn']]` |
| decode | `model.layers.*.shared_transformer.self_attn.linear_q_adapter_list.*.*` | matmul | `[['B', 'd_attn'], ['d_attn', 'r_lora']]` | `['r_lora', 'd_attn']` | `[['B', 'r_lora']]` |
| decode | `model.layers.*.shared_transformer.self_attn.linear_q_adapter_list.*.*` | matmul | `[['B', 'r_lora'], ['r_lora', 'd_attn']]` | `['d_attn', 'r_lora']` | `[['B', 'd_attn']]` |
| decode | `model.layers.*.shared_transformer.self_attn.linear_k_adapter_list.*.*` | matmul | `[['B', 'd_attn'], ['d_attn', 'r_lora']]` | `['r_lora', 'd_attn']` | `[['B', 'r_lora']]` |
| decode | `model.layers.*.shared_transformer.self_attn.linear_k_adapter_list.*.*` | matmul | `[['B', 'r_lora'], ['r_lora', 'd_attn']]` | `['d_attn', 'r_lora']` | `[['B', 'd_attn']]` |
| decode | `model.layers.*.shared_transformer.self_attn.linear_v_adapter_list.*.*` | matmul | `[['B', 'd_attn'], ['d_attn', 'r_lora']]` | `['r_lora', 'd_attn']` | `[['B', 'r_lora']]` |
| decode | `model.layers.*.shared_transformer.self_attn.linear_v_adapter_list.*.*` | matmul | `[['B', 'r_lora'], ['r_lora', 'd_attn']]` | `['d_attn', 'r_lora']` | `[['B', 'd_attn']]` |
| decode | `model.layers.*.shared_transformer.self_attn` | batched_matmul | `[['n_h', 'B', 'd_head'], ['n_h', 'd_head', 'T+1']]` | `None` | `[['n_h', 'B', 'T+1']]` |
| decode | `model.layers.*.shared_transformer.self_attn` | softmax | `[['B', 'n_h', '1', 'T+1']]` | `None` | `[['B', 'n_h', '1', 'T+1']]` |
| decode | `model.layers.*.shared_transformer.self_attn` | batched_matmul | `[['n_h', 'B', 'T+1'], ['n_h', 'T+1', 'd_head']]` | `None` | `[['n_h', 'B', 'd_head']]` |
| decode | `model.layers.*.shared_transformer.self_attn.o_proj` | matmul | `[['B', 'd_attn'], ['d_attn', 'd_model']]` | `['d_model', 'd_attn']` | `[['B', 'd_model']]` |
| decode | `model.layers.*.shared_transformer.pre_ff_layernorm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.shared_transformer.feed_forward.gate_up_proj` | matmul | `[['B', 'd_model'], ['d_model', '2*d_ff']]` | `['2*d_ff', 'd_model']` | `[['B', '2*d_ff']]` |
| decode | `model.layers.*.shared_transformer.feed_forward.gate_up_proj_adapter_list.*.*` | matmul | `[['B', 'd_model'], ['d_model', 'r_lora']]` | `['r_lora', 'd_model']` | `[['B', 'r_lora']]` |
| decode | `model.layers.*.shared_transformer.feed_forward.gate_up_proj_adapter_list.*.*` | matmul | `[['B', 'r_lora'], ['r_lora', '2*d_ff']]` | `['2*d_ff', 'r_lora']` | `[['B', '2*d_ff']]` |
| decode | `model.layers.*.shared_transformer.feed_forward` | elementwise_add | `[['B', '1', '2*d_ff'], ['B', '1', '2*d_ff']]` | `None` | `[['B', '1', '2*d_ff']]` |
| decode | `model.layers.*.shared_transformer.feed_forward.act_fn` | gelu | `[['B', '1', 'd_ff']]` | `None` | `[['B', '1', 'd_ff']]` |
| decode | `model.layers.*.shared_transformer.feed_forward` | elementwise_mul | `[['B', '1', 'd_ff'], ['B', '1', 'd_ff']]` | `None` | `[['B', '1', 'd_ff']]` |
| decode | `model.layers.*.shared_transformer.feed_forward.down_proj` | matmul | `[['B', 'd_ff'], ['d_ff', 'd_model']]` | `['d_model', 'd_ff']` | `[['B', 'd_model']]` |
| decode | `model.layers.*.linear` | matmul | `[['B', 'd_model'], ['d_model', 'd_model']]` | `['d_model', 'd_model']` | `[['B', 'd_model']]` |
| decode | `model.layers.*.mamba_decoder` | elementwise_add | `[['B', '1', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.mamba_decoder.input_layernorm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.mamba_decoder.mamba.in_proj` | matmul | `[['B', 'd_model'], ['d_model', '2*d_inner+2*n_g*d_state+n_h_ssm']]` | `['2*d_inner+2*n_g*d_state+n_h_ssm', 'd_model']` | `[['B', '2*d_inner+2*n_g*d_state+n_h_ssm']]` |
| decode | `model.layers.*.mamba_decoder.mamba.act` | silu | `[['B', 'd_inner+2*n_g*d_state']]` | `None` | `[['B', 'd_inner+2*n_g*d_state']]` |
| decode | `model.layers.*.mamba_decoder.mamba` | exp | `[['d_head_ssm']]` | `None` | `[['d_head_ssm']]` |
| decode | `model.layers.*.mamba_decoder.mamba` | exp | `[['B', 'd_head_ssm', 'n_h_ssm', 'd_state']]` | `None` | `[['B', 'd_head_ssm', 'n_h_ssm', 'd_state']]` |
| decode | `model.layers.*.mamba_decoder.mamba` | batched_matmul | `[['d_head_ssm', 'n_h_ssm', 'd_state'], ['d_head_ssm', 'd_state', 'B']]` | `None` | `[['d_head_ssm', 'n_h_ssm', 'B']]` |
| decode | `model.layers.*.mamba_decoder.mamba.norm` | rmsnorm | `[['B', '1', 'd_inner']]` | `['d_inner']` | `[['B', '1', 'd_inner']]` |
| decode | `model.layers.*.mamba_decoder.mamba.out_proj` | matmul | `[['B', 'd_inner'], ['d_inner', 'd_model']]` | `['d_model', 'd_inner']` | `[['B', 'd_model']]` |
| decode | `model.final_layernorm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `lm_head` | matmul | `[['B', 'd_model'], ['d_model', 'V']]` | `['V', 'd_model']` | `[['B', 'V']]` |

## 전수 점검 — 이 모델이 쓰는 이름 전부

위 절이 '풀리지 않은 것'이라면 여기는 **전부**다. 규칙이 자신 있게 붙인 이름도 틀릴 수 있고, 그런 건 미결 목록에 절대 오르지 않는다. 한 줄씩 읽고 **그 모듈에서 그 이름이 말이 되는지** 보라.

### A. 붙은 이름 전부 (22종)

| 라벨 | 값 | 나타나는 모듈 | 축 수 |
|---|---|---|---|
| `B` |  | `model.layers.*.mamba`, `model.layers.*.mamba_decoder.mamba`, `model.layers.*.mamba.norm`, `model.layers.*.shared_transformer.self_attn` 외 70개 | 19666 |
| `d_head_ssm` | 64 | `model.layers.*.mamba`, `model.layers.*.mamba_decoder.mamba` | 11210 |
| `d_chunk` | 256 | `model.layers.*.mamba`, `model.layers.*.mamba_decoder.mamba` | 6118 |
| `T` |  | `model.layers.*.mamba`, `model.layers.*.shared_transformer.self_attn`, `model.layers.*.mamba.norm`, `model.layers.*.input_layernorm` 외 70개 | 5902 |
| `d_state` | 128 | `model.layers.*.mamba`, `model.layers.*.mamba_decoder.mamba` | 5092 |
| `n_h_ssm` | 64 | `model.layers.*.mamba`, `model.layers.*.mamba_decoder.mamba` | 4712 |
| `d_model` | 2048 | `model.layers.*.input_layernorm`, `model.layers.*.mamba.in_proj`, `model.layers.*.mamba.out_proj`, `model.layers.*.linear` 외 47개 | 2830 |
| `d_inner` |  | `model.layers.*.mamba.norm`, `model.layers.*.mamba.out_proj`, `model.layers.*.mamba`, `model.layers.*.mamba_decoder.mamba.norm` 외 2개 | 2280 |
| `d_inner+2*n_g*d_state` |  | `model.layers.*.mamba`, `model.layers.*.mamba_decoder.mamba`, `model.layers.*.mamba.conv1d`, `model.layers.*.mamba.act` 외 2개 | 2052 |
| `d_attn` | 4096 | `model.layers.*.shared_transformer.self_attn.q_proj`, `model.layers.*.shared_transformer.self_attn.k_proj`, `model.layers.*.shared_transformer.self_attn.v_proj`, `model.layers.*.shared_transformer.self_attn` 외 9개 | 1512 |
| `n_h` | 32 | `model.layers.*.shared_transformer.self_attn` | 1020 |
| `d_conv` | 4 | `model.layers.*.mamba`, `model.layers.*.mamba_decoder.mamba`, `model.layers.*.mamba.conv1d`, `model.layers.*.mamba_decoder.mamba.conv1d` | 912 |
| `d_head` | 128 | `model.layers.*.shared_transformer.self_attn`, `model.rotary_emb` | 858 |
| `r_lora` | 128 | `model.layers.*.shared_transformer.self_attn.linear_q_adapter_list.*.0`, `model.layers.*.shared_transformer.self_attn.linear_q_adapter_list.*.1`, `model.layers.*.shared_transformer.self_attn.linear_k_adapter_list.*.0`, `model.layers.*.shared_transformer.self_attn.linear_k_adapter_list.*.1` 외 4개 | 768 |
| `2*d_inner+2*n_g*d_state+n_h_ssm` |  | `model.layers.*.mamba.in_proj`, `model.layers.*.mamba_decoder.mamba.in_proj`, `model.layers.*.mamba`, `model.layers.*.mamba_decoder.mamba` | 684 |
| `2*d_ff` |  | `model.layers.*.shared_transformer.feed_forward.gate_up_proj`, `model.layers.*.shared_transformer.feed_forward.gate_up_proj_adapter_list.*.1`, `model.layers.*.shared_transformer.feed_forward` | 240 |
| `d_head/2` |  | `model.layers.*.shared_transformer.self_attn`, `model.rotary_emb` | 180 |
| `d_ff` | 8192 | `model.layers.*.shared_transformer.feed_forward.down_proj`, `model.layers.*.shared_transformer.feed_forward`, `model.layers.*.shared_transformer.feed_forward.act_fn` | 180 |
| `T+1` |  | `model.layers.*.shared_transformer.self_attn` | 156 |
| `d_conv+1` |  | `model.layers.*.mamba`, `model.layers.*.mamba_decoder.mamba` | 114 |
| `T+d_conv-1` |  | `model.layers.*.mamba.conv1d`, `model.layers.*.mamba`, `model.layers.*.mamba_decoder.mamba.conv1d`, `model.layers.*.mamba_decoder.mamba` | 76 |
| `V` | 32000 | `lm_head`, `model.embed_tokens` | 20 |

### B. 이름 없이 남은 정수 전부 (2쌍)

**여기가 필터가 못 보던 자리다.** 정수가 남는 것 자체는 정상이다(루프 인덱스, 피연산자 개수, 브로드캐스트 축). 문제는 **이름이 있어야 하는데 없는 경우**이고, 마지막 열이 그 신호다 — 이 모델의 심볼과 값이 같다면 스코프가 그 모듈을 못 덮고 있을 수 있다. 실제로 `n_hc`(=4)가 그렇게 정수로 남아 있었다.

| 모듈 | 정수 | 축 수 | 같은 값의 심볼 |
|---|---|---|---|
| `model.layers.*.mamba` | 2 | 2272 | — |
| `model.layers.*.mamba_decoder.mamba` | 2 | 426 | — |

### C. 모듈이 내는 출력 shape 전부 (74개 모듈 / 443종)

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
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.embed_tokens`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.final_layernorm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
- `model.layers.*.input_layernorm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
- `model.layers.*.linear`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[T, d_model]]`
  - `[[d_model, d_model]]`
- `model.layers.*.mamba`
  - `[[2, 2]]`
  - `[[B, 1, 0], [B, 1, 0], [B, 1, d_inner], [B, 1, d_inner+2*n_g*d_state], [B, 1, d_head_ssm]]`
  - `[[B, 1, 1, d_chunk, d_head_ssm, d_state]]`
  - `[[B, 1, 1, d_chunk, d_head_ssm, n_h_ssm]]`
  - `[[B, 1, 1, d_head_ssm, n_h_ssm, d_state]]`
  - `[[B, 1, 1, d_state]]`
  - `[[B, 1, d_chunk, 1, d_head_ssm, d_state]]`
  - `[[B, 1, d_chunk, d_chunk, d_head_ssm, 1]]`
  - `[[B, 1, d_chunk, d_chunk, d_head_ssm, d_state]]`
  - `[[B, 1, d_chunk, d_chunk, d_head_ssm, n_h_ssm]]`
  - `[[B, 1, d_chunk, d_chunk, d_head_ssm]]`
  - `[[B, 1, d_chunk, d_head_ssm, 1, d_state]]`
  - `[[B, 1, d_chunk, d_head_ssm, 1]]`
  - `[[B, 1, d_chunk, d_head_ssm, d_state]]`
  - `[[B, 1, d_chunk, d_head_ssm, n_h_ssm, d_state]]`
  - `[[B, 1, d_chunk, d_head_ssm, n_h_ssm]]`
  - `[[B, 1, d_chunk, d_head_ssm]]`
  - `[[B, 1, d_head_ssm, d_chunk, 1, n_h_ssm]]`
  - `[[B, 1, d_head_ssm, d_chunk, d_state, 1]]`
  - `[[B, 1, d_head_ssm, d_chunk, d_state, n_h_ssm]]`
  - `[[B, 1, d_head_ssm, d_chunk, d_state]]`
  - `[[B, 1, d_head_ssm, d_chunk, n_h_ssm]]`
  - `[[B, 1, d_head_ssm, d_state, n_h_ssm]]`
  - `[[B, 1, d_head_ssm, d_state]]`
  - `[[B, 1, d_head_ssm, n_h_ssm, d_state]]`
  - `[[B, 1, d_head_ssm]]`
  - `[[B, 1, d_inner+2*n_g*d_state]]`
  - `[[B, 1, d_inner], [B, 1, d_state], [B, 1, d_state]]`
  - `[[B, 1, d_inner]]`
  - `[[B, 1, d_state]]`
  - `[[B, 2, d_head_ssm, n_h_ssm, d_state]]`
  - `[[B, T, 0], [B, T, 0], [B, T, d_inner], [B, T, d_inner+2*n_g*d_state], [B, T, d_head_ssm]]`
  - `[[B, T, 1, 1, d_state]]`
  - `[[B, T, 1, d_head_ssm, d_state]]`
  - `[[B, T, 1, d_state]]`
  - `[[B, T, d_head_ssm, 1]]`
  - `[[B, T, d_head_ssm, d_state]]`
  - `[[B, T, d_head_ssm, n_h_ssm]]`
  - `[[B, T, d_head_ssm]]`
  - `[[B, T, d_inner+2*n_g*d_state]]`
  - `[[B, T, d_inner], [B, T, d_state], [B, T, d_state]]`
  - `[[B, T, d_inner]]`
  - `[[B, d_chunk, d_head_ssm, d_state]]`
  - `[[B, d_chunk, d_head_ssm, n_h_ssm]]`
  - `[[B, d_chunk, d_head_ssm]]`
  - `[[B, d_head_ssm, 1, 1]]`
  - `[[B, d_head_ssm, 1, 2, n_h_ssm, d_state]]`
  - `[[B, d_head_ssm, 1, d_chunk, 1]]`
  - `[[B, d_head_ssm, 1, d_chunk, d_chunk]]`
  - `[[B, d_head_ssm, 1, d_chunk]]`
  - `[[B, d_head_ssm, 1, d_state]]`
  - `[[B, d_head_ssm, 1]]`
  - `[[B, d_head_ssm, 2, 1]]`
  - `[[B, d_head_ssm, 2, 2, 1, 1]]`
  - `[[B, d_head_ssm, 2, 2, 1]]`
  - `[[B, d_head_ssm, 2, 2, n_h_ssm, d_state]]`
  - `[[B, d_head_ssm, 2, 2]]`
  - `[[B, d_head_ssm, 2, n_h_ssm, d_state]]`
  - `[[B, d_head_ssm, 2]]`
  - `[[B, d_head_ssm, d_state]]`
  - `[[B, d_head_ssm, n_h_ssm, 1]]`
  - `[[B, d_head_ssm, n_h_ssm, d_state]]`
  - `[[B, d_head_ssm, n_h_ssm]]`
  - `[[B, d_head_ssm]]`
  - `[[B, d_inner+2*n_g*d_state, 1]]`
  - `[[B, d_inner+2*n_g*d_state, T]]`
  - `[[B, d_inner+2*n_g*d_state, d_conv+1]]`
  - `[[B, d_inner+2*n_g*d_state, d_conv]]`
  - `[[B, d_inner+2*n_g*d_state]]`
  - `[[B, d_inner]]`
  - `[[d_chunk, d_chunk]]`
  - `[[d_head_ssm, B, 1]]`
  - `[[d_head_ssm, B]]`
  - `[[d_head_ssm, d_state, B]]`
  - `[[d_head_ssm, n_h_ssm, B]]`
  - `[[d_head_ssm, n_h_ssm, d_state]]`
  - `[[d_head_ssm, n_h_ssm]]`
  - `[[d_head_ssm]]`
  - `[[d_inner+2*n_g*d_state, d_conv]]`
- `model.layers.*.mamba.act`
  - `[[B, T, d_inner+2*n_g*d_state]]`
  - `[[B, d_inner+2*n_g*d_state]]`
- `model.layers.*.mamba.conv1d`
  - `[[B, d_inner+2*n_g*d_state, T+d_conv-1]]`
- `model.layers.*.mamba.in_proj`
  - `[[B, 1, 2*d_inner+2*n_g*d_state+n_h_ssm]]`
  - `[[B, 2*d_inner+2*n_g*d_state+n_h_ssm]]`
  - `[[B, T, 2*d_inner+2*n_g*d_state+n_h_ssm]]`
  - `[[B, d_model]]`
  - `[[T, 2*d_inner+2*n_g*d_state+n_h_ssm]]`
  - `[[T, d_model]]`
  - `[[d_model, 2*d_inner+2*n_g*d_state+n_h_ssm]]`
- `model.layers.*.mamba.norm`
  - `[[B, 1, 1, 1]]`
  - `[[B, 1, 1, d_inner]]`
  - `[[B, 1, d_inner]]`
  - `[[B, T, 1, 1]]`
  - `[[B, T, 1, d_inner]]`
  - `[[B, T, d_inner]]`
- `model.layers.*.mamba.out_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_inner]]`
  - `[[B, d_model]]`
  - `[[T, d_inner]]`
  - `[[T, d_model]]`
  - `[[d_inner, d_model]]`
- `model.layers.*.mamba_decoder`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.*.mamba_decoder.input_layernorm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
- `model.layers.*.mamba_decoder.mamba`
  - `[[2, 2]]`
  - `[[B, 1, 0], [B, 1, 0], [B, 1, d_inner], [B, 1, d_inner+2*n_g*d_state], [B, 1, d_head_ssm]]`
  - `[[B, 1, 1, d_chunk, d_head_ssm, d_state]]`
  - `[[B, 1, 1, d_chunk, d_head_ssm, n_h_ssm]]`
  - `[[B, 1, 1, d_head_ssm, n_h_ssm, d_state]]`
  - `[[B, 1, 1, d_state]]`
  - `[[B, 1, d_chunk, 1, d_head_ssm, d_state]]`
  - `[[B, 1, d_chunk, d_chunk, d_head_ssm, 1]]`
  - `[[B, 1, d_chunk, d_chunk, d_head_ssm, d_state]]`
  - `[[B, 1, d_chunk, d_chunk, d_head_ssm, n_h_ssm]]`
  - `[[B, 1, d_chunk, d_chunk, d_head_ssm]]`
  - `[[B, 1, d_chunk, d_head_ssm, 1, d_state]]`
  - `[[B, 1, d_chunk, d_head_ssm, 1]]`
  - `[[B, 1, d_chunk, d_head_ssm, d_state]]`
  - `[[B, 1, d_chunk, d_head_ssm, n_h_ssm, d_state]]`
  - `[[B, 1, d_chunk, d_head_ssm, n_h_ssm]]`
  - `[[B, 1, d_chunk, d_head_ssm]]`
  - `[[B, 1, d_head_ssm, d_chunk, 1, n_h_ssm]]`
  - `[[B, 1, d_head_ssm, d_chunk, d_state, 1]]`
  - `[[B, 1, d_head_ssm, d_chunk, d_state, n_h_ssm]]`
  - `[[B, 1, d_head_ssm, d_chunk, d_state]]`
  - `[[B, 1, d_head_ssm, d_chunk, n_h_ssm]]`
  - `[[B, 1, d_head_ssm, d_state, n_h_ssm]]`
  - `[[B, 1, d_head_ssm, d_state]]`
  - `[[B, 1, d_head_ssm, n_h_ssm, d_state]]`
  - `[[B, 1, d_head_ssm]]`
  - `[[B, 1, d_inner+2*n_g*d_state]]`
  - `[[B, 1, d_inner], [B, 1, d_state], [B, 1, d_state]]`
  - `[[B, 1, d_inner]]`
  - `[[B, 1, d_state]]`
  - `[[B, 2, d_head_ssm, n_h_ssm, d_state]]`
  - `[[B, T, 0], [B, T, 0], [B, T, d_inner], [B, T, d_inner+2*n_g*d_state], [B, T, d_head_ssm]]`
  - `[[B, T, 1, 1, d_state]]`
  - `[[B, T, 1, d_head_ssm, d_state]]`
  - `[[B, T, 1, d_state]]`
  - `[[B, T, d_head_ssm, 1]]`
  - `[[B, T, d_head_ssm, d_state]]`
  - `[[B, T, d_head_ssm, n_h_ssm]]`
  - `[[B, T, d_head_ssm]]`
  - `[[B, T, d_inner+2*n_g*d_state]]`
  - `[[B, T, d_inner], [B, T, d_state], [B, T, d_state]]`
  - `[[B, T, d_inner]]`
  - `[[B, d_chunk, d_head_ssm, d_state]]`
  - `[[B, d_chunk, d_head_ssm, n_h_ssm]]`
  - `[[B, d_chunk, d_head_ssm]]`
  - `[[B, d_head_ssm, 1, 1]]`
  - `[[B, d_head_ssm, 1, 2, n_h_ssm, d_state]]`
  - `[[B, d_head_ssm, 1, d_chunk, 1]]`
  - `[[B, d_head_ssm, 1, d_chunk, d_chunk]]`
  - `[[B, d_head_ssm, 1, d_chunk]]`
  - `[[B, d_head_ssm, 1, d_state]]`
  - `[[B, d_head_ssm, 1]]`
  - `[[B, d_head_ssm, 2, 1]]`
  - `[[B, d_head_ssm, 2, 2, 1, 1]]`
  - `[[B, d_head_ssm, 2, 2, 1]]`
  - `[[B, d_head_ssm, 2, 2, n_h_ssm, d_state]]`
  - `[[B, d_head_ssm, 2, 2]]`
  - `[[B, d_head_ssm, 2, n_h_ssm, d_state]]`
  - `[[B, d_head_ssm, 2]]`
  - `[[B, d_head_ssm, d_state]]`
  - `[[B, d_head_ssm, n_h_ssm, 1]]`
  - `[[B, d_head_ssm, n_h_ssm, d_state]]`
  - `[[B, d_head_ssm, n_h_ssm]]`
  - `[[B, d_head_ssm]]`
  - `[[B, d_inner+2*n_g*d_state, 1]]`
  - `[[B, d_inner+2*n_g*d_state, T]]`
  - `[[B, d_inner+2*n_g*d_state, d_conv+1]]`
  - `[[B, d_inner+2*n_g*d_state, d_conv]]`
  - `[[B, d_inner+2*n_g*d_state]]`
  - `[[B, d_inner]]`
  - `[[d_chunk, d_chunk]]`
  - `[[d_head_ssm, B, 1]]`
  - `[[d_head_ssm, B]]`
  - `[[d_head_ssm, d_state, B]]`
  - `[[d_head_ssm, n_h_ssm, B]]`
  - `[[d_head_ssm, n_h_ssm, d_state]]`
  - `[[d_head_ssm, n_h_ssm]]`
  - `[[d_head_ssm]]`
  - `[[d_inner+2*n_g*d_state, d_conv]]`
- `model.layers.*.mamba_decoder.mamba.act`
  - `[[B, T, d_inner+2*n_g*d_state]]`
  - `[[B, d_inner+2*n_g*d_state]]`
- `model.layers.*.mamba_decoder.mamba.conv1d`
  - `[[B, d_inner+2*n_g*d_state, T+d_conv-1]]`
- `model.layers.*.mamba_decoder.mamba.in_proj`
  - `[[B, 1, 2*d_inner+2*n_g*d_state+n_h_ssm]]`
  - `[[B, 2*d_inner+2*n_g*d_state+n_h_ssm]]`
  - `[[B, T, 2*d_inner+2*n_g*d_state+n_h_ssm]]`
  - `[[B, d_model]]`
  - `[[T, 2*d_inner+2*n_g*d_state+n_h_ssm]]`
  - `[[T, d_model]]`
  - `[[d_model, 2*d_inner+2*n_g*d_state+n_h_ssm]]`
- `model.layers.*.mamba_decoder.mamba.norm`
  - `[[B, 1, 1, 1]]`
  - `[[B, 1, 1, d_inner]]`
  - `[[B, 1, d_inner]]`
  - `[[B, T, 1, 1]]`
  - `[[B, T, 1, d_inner]]`
  - `[[B, T, d_inner]]`
- `model.layers.*.mamba_decoder.mamba.out_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_inner]]`
  - `[[B, d_model]]`
  - `[[T, d_inner]]`
  - `[[T, d_model]]`
  - `[[d_inner, d_model]]`
- `model.layers.*.shared_transformer`
  - `[[B, 1, d_attn]]`
  - `[[B, T, d_attn]]`
- `model.layers.*.shared_transformer.feed_forward`
  - `[[B, 1, 2*d_ff]]`
  - `[[B, 1, d_ff], [B, 1, d_ff]]`
  - `[[B, 1, d_ff]]`
  - `[[B, T, 2*d_ff]]`
  - `[[B, T, d_ff], [B, T, d_ff]]`
  - `[[B, T, d_ff]]`
- `model.layers.*.shared_transformer.feed_forward.act_fn`
  - `[[B, 1, d_ff]]`
  - `[[B, T, d_ff]]`
- `model.layers.*.shared_transformer.feed_forward.down_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_ff]]`
  - `[[B, d_model]]`
  - `[[T, d_ff]]`
  - `[[T, d_model]]`
  - `[[d_ff, d_model]]`
- `model.layers.*.shared_transformer.feed_forward.gate_up_proj`
  - `[[B, 1, 2*d_ff]]`
  - `[[B, 2*d_ff]]`
  - `[[B, T, 2*d_ff]]`
  - `[[B, d_model]]`
  - `[[T, 2*d_ff]]`
  - `[[T, d_model]]`
  - `[[d_model, 2*d_ff]]`
- `model.layers.*.shared_transformer.feed_forward.gate_up_proj_adapter_list.*.0`
  - `[[B, 1, r_lora]]`
  - `[[B, T, r_lora]]`
  - `[[B, d_model]]`
  - `[[B, r_lora]]`
  - `[[T, d_model]]`
  - `[[T, r_lora]]`
  - `[[d_model, r_lora]]`
- `model.layers.*.shared_transformer.feed_forward.gate_up_proj_adapter_list.*.1`
  - `[[B, 1, 2*d_ff]]`
  - `[[B, 2*d_ff]]`
  - `[[B, T, 2*d_ff]]`
  - `[[B, r_lora]]`
  - `[[T, 2*d_ff]]`
  - `[[T, r_lora]]`
  - `[[r_lora, 2*d_ff]]`
- `model.layers.*.shared_transformer.input_layernorm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_attn]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_attn]]`
- `model.layers.*.shared_transformer.pre_ff_layernorm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
- `model.layers.*.shared_transformer.self_attn`
  - `[[B, 1, 1, d_head]]`
  - `[[B, 1, T, d_head]]`
  - `[[B, 1, d_attn]]`
  - `[[B, 1, n_h, d_head]]`
  - `[[B, T, d_attn]]`
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
  - `[[T, T]]`
  - `[[]]`
  - `[[n_h, B, T+1]]`
  - `[[n_h, B, d_head]]`
  - `[[n_h, T+1, d_head]]`
  - `[[n_h, T, T]]`
  - `[[n_h, T, d_head]]`
  - `[[n_h, d_head, T+1]]`
  - `[[n_h, d_head, T]]`
- `model.layers.*.shared_transformer.self_attn.k_proj`
  - `[[B, 1, d_attn]]`
  - `[[B, T, d_attn]]`
  - `[[B, d_attn]]`
  - `[[T, d_attn]]`
  - `[[d_attn, d_attn]]`
- `model.layers.*.shared_transformer.self_attn.linear_k_adapter_list.*.0`
  - `[[B, 1, r_lora]]`
  - `[[B, T, r_lora]]`
  - `[[B, d_attn]]`
  - `[[B, r_lora]]`
  - `[[T, d_attn]]`
  - `[[T, r_lora]]`
  - `[[d_attn, r_lora]]`
- `model.layers.*.shared_transformer.self_attn.linear_k_adapter_list.*.1`
  - `[[B, 1, d_attn]]`
  - `[[B, T, d_attn]]`
  - `[[B, d_attn]]`
  - `[[B, r_lora]]`
  - `[[T, d_attn]]`
  - `[[T, r_lora]]`
  - `[[r_lora, d_attn]]`
- `model.layers.*.shared_transformer.self_attn.linear_q_adapter_list.*.0`
  - `[[B, 1, r_lora]]`
  - `[[B, T, r_lora]]`
  - `[[B, d_attn]]`
  - `[[B, r_lora]]`
  - `[[T, d_attn]]`
  - `[[T, r_lora]]`
  - `[[d_attn, r_lora]]`
- `model.layers.*.shared_transformer.self_attn.linear_q_adapter_list.*.1`
  - `[[B, 1, d_attn]]`
  - `[[B, T, d_attn]]`
  - `[[B, d_attn]]`
  - `[[B, r_lora]]`
  - `[[T, d_attn]]`
  - `[[T, r_lora]]`
  - `[[r_lora, d_attn]]`
- `model.layers.*.shared_transformer.self_attn.linear_v_adapter_list.*.0`
  - `[[B, 1, r_lora]]`
  - `[[B, T, r_lora]]`
  - `[[B, d_attn]]`
  - `[[B, r_lora]]`
  - `[[T, d_attn]]`
  - `[[T, r_lora]]`
  - `[[d_attn, r_lora]]`
- `model.layers.*.shared_transformer.self_attn.linear_v_adapter_list.*.1`
  - `[[B, 1, d_attn]]`
  - `[[B, T, d_attn]]`
  - `[[B, d_attn]]`
  - `[[B, r_lora]]`
  - `[[T, d_attn]]`
  - `[[T, r_lora]]`
  - `[[r_lora, d_attn]]`
- `model.layers.*.shared_transformer.self_attn.o_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_attn]]`
  - `[[B, d_model]]`
  - `[[T, d_attn]]`
  - `[[T, d_model]]`
  - `[[d_attn, d_model]]`
- `model.layers.*.shared_transformer.self_attn.q_proj`
  - `[[B, 1, d_attn]]`
  - `[[B, T, d_attn]]`
  - `[[B, d_attn]]`
  - `[[T, d_attn]]`
  - `[[d_attn, d_attn]]`
- `model.layers.*.shared_transformer.self_attn.v_proj`
  - `[[B, 1, d_attn]]`
  - `[[B, T, d_attn]]`
  - `[[B, d_attn]]`
  - `[[T, d_attn]]`
  - `[[d_attn, d_attn]]`
- `model.layers.0`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.1`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.10`
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
- `model.layers.36`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.37`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.4`
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
