# 검토 의뢰서 — moonshotai/Kimi-K2.6

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `kimi_k2`
- 판단 필요: **0건**

## 증거 — 이미 받아둔 실제 소스

- `develop/sources/modeling_kimi_k2.py` — **없음** (네트워크 불가 또는 transformers 본체에 없는 아키텍처)
- `develop/sources/configuration_kimi_k2.py` — **없음** (네트워크 불가 또는 transformers 본체에 없는 아키텍처)

- 온라인 원본: https://github.com/huggingface/transformers/tree/main/src/transformers/models/kimi_k2

그 밖의 재료: `full/review.md`(리뷰 패킷 — shape 별 실제 행 표본), `structure.yaml`(이 모델의 심볼 표), `full/<phase>.csv`(전체 operator 표).

## 판단이 필요한 것

없다. 이 모델의 축은 전부 등록된 규칙이 이름을 냈고, 소스 대조도 어긋난 곳이 없다.

그래도 검토를 돌린다면 `full/review.md` 의 표본을 보고 규칙 자체가 틀리지 않았는지를 본다 — 그것이 규칙 게이트가 구조적으로 못 보는 부분이다.

## 기계적으로 이미 확인된 것 — 다시 묻지 말 것

- **심볼이 읽은 config 필드**: 전부 이 모델의 config 클래스(또는 상속/프로퍼티/getattr 기본값)에 존재한다
- **정사각 축**: 소스에서 정사각 생성/reshape 과 대응이 확인된 축 없음
- **모듈이 읽는 config 속성**: `__init__` 에서 config 를 읽는 클래스 0개를 소스에서 확인했다. 그 목록이 각 모듈의 폭이 가질 수 있는 이름의 전부다.

## 전수 점검 — 이 모델이 쓰는 이름 전부

위 절이 '풀리지 않은 것'이라면 여기는 **전부**다. 규칙이 자신 있게 붙인 이름도 틀릴 수 있고, 그런 건 미결 목록에 절대 오르지 않는다. 한 줄씩 읽고 **그 모듈에서 그 이름이 말이 되는지** 보라.

### A. 붙은 이름 전부 (24종)

| 라벨 | 값 | 나타나는 모듈 | 축 수 |
|---|---|---|---|
| `B` |  | `model.layers.*.self_attn`, `model.layers.*.mlp.gate`, `model.layers.*.input_layernorm`, `model.layers.*.self_attn.q_a_layernorm` 외 84개 | 34814 |
| `T` |  | `model.layers.*.self_attn`, `model.layers.*.mlp.gate`, `model.layers.*.input_layernorm`, `model.layers.*.self_attn.q_a_layernorm` 외 84개 | 22192 |
| `d_model` | 7168 | `model.layers.*.mlp.experts`, `model.layers.*.mlp.gate`, `model.layers.*.input_layernorm`, `model.layers.*.post_attention_layernorm` 외 75개 | 15326 |
| `n_h` | 64 | `model.layers.*.self_attn` | 12566 |
| `d_rope/2` |  | `model.layers.*.self_attn`, `model.rotary_emb` | 6136 |
| `E` | 384 | `model.layers.*.mlp.gate`, `model.layers.*.mlp.experts` | 5280 |
| `d_moe` | 2048 | `model.layers.*.mlp.experts`, `model.layers.*.mlp.shared_experts.gate_proj`, `model.layers.*.mlp.shared_experts.up_proj`, `model.layers.*.mlp.shared_experts.down_proj` 외 3개 | 5040 |
| `k` | 8 | `model.layers.*.mlp.experts`, `model.layers.*.mlp.gate`, `model.layers.*.mlp.experts.act_fn` | 4860 |
| `d_nope+d_rope` |  | `model.layers.*.self_attn` | 3477 |
| `c_q` | 1536 | `model.layers.*.self_attn.q_a_layernorm`, `model.layers.*.self_attn.q_a_proj`, `model.layers.*.self_attn.q_b_proj` | 3416 |
| `k*T` |  | `model.layers.*.mlp.experts`, `model.layers.*.mlp.experts.act_fn` | 3300 |
| `d_nope` | 128 | `model.layers.*.self_attn` | 2867 |
| `c_kv` | 512 | `model.layers.*.self_attn.kv_a_layernorm`, `model.layers.*.self_attn.kv_b_proj`, `model.layers.*.self_attn` | 2562 |
| `T+1` |  | `model.layers.*.self_attn` | 1952 |
| `(n_h+2*n_kv)*d_head` |  | `model.layers.*.self_attn.q_b_proj`, `model.layers.*.self_attn` | 1098 |
| `c_kv+d_rope` |  | `model.layers.*.self_attn.kv_a_proj_with_mqa`, `model.layers.*.self_attn` | 1098 |
| `n_h*(d_nope+d_v)` |  | `model.layers.*.self_attn.kv_b_proj`, `model.layers.*.self_attn` | 1098 |
| `n_h*d_v` |  | `model.layers.*.self_attn.o_proj`, `model.layers.*.self_attn` | 1098 |
| `d_head` | 64 | `model.layers.*.self_attn`, `model.rotary_emb` | 880 |
| `2*d_moe` |  | `model.layers.*.mlp.experts` | 840 |
| `d_nope+d_v` |  | `model.layers.*.self_attn` | 244 |
| `2*d_nope` |  | `model.layers.*.self_attn` | 244 |
| `d_ff` | 18432 | `model.layers.*.mlp.gate_proj`, `model.layers.*.mlp.up_proj`, `model.layers.*.mlp.down_proj`, `model.layers.*.mlp` 외 1개 | 58 |
| `V` | 163840 | `lm_head`, `model.embed_tokens` | 20 |

### B. 이름 없이 남은 정수 전부 (1쌍)

**여기가 필터가 못 보던 자리다.** 정수가 남는 것 자체는 정상이다(루프 인덱스, 피연산자 개수, 브로드캐스트 축). 문제는 **이름이 있어야 하는데 없는 경우**이고, 마지막 열이 그 신호다 — 이 모델의 심볼과 값이 같다면 스코프가 그 모듈을 못 덮고 있을 수 있다. 실제로 `n_hc`(=4)가 그렇게 정수로 남아 있었다.

| 모듈 | 정수 | 축 수 | 같은 값의 심볼 |
|---|---|---|---|
| `model.layers.*.mlp.gate` | 2 | 360 | — |

### C. 모듈이 내는 출력 shape 전부 (89개 모듈 / 341종)

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
  - `[[B, d_model]]`
  - `[[B, k, d_model]]`
  - `[[E, d_model, 2*d_moe]]`
  - `[[E, d_moe, d_model]]`
  - `[[E]]`
  - `[[T, d_model]]`
  - `[[T, k, d_model]]`
  - `[[k*T, 2*d_moe]]`
  - `[[k*T, B]]`
  - `[[k*T, d_model]]`
  - `[[k*T, d_moe], [k*T, d_moe]]`
  - `[[k*T, d_moe]]`
  - `[[k*T], [k*T]]`
  - `[[k*T]]`
  - `[[k, 2*d_moe]]`
  - `[[k, B]]`
  - `[[k, d_model]]`
  - `[[k, d_moe], [k, d_moe]]`
  - `[[k, d_moe]]`
  - `[[k], [k]]`
  - `[[k]]`
- `model.layers.*.mlp.experts.act_fn`
  - `[[k*T, d_moe]]`
  - `[[k, d_moe]]`
- `model.layers.*.mlp.gate`
  - `[[B, 1, 1]]`
  - `[[B, 1, 2], [B, 1, 2]]`
  - `[[B, 1, E]]`
  - `[[B, 1], [B, 1]]`
  - `[[B, 1]]`
  - `[[B, E]]`
  - `[[B, d_model]]`
  - `[[B, k], [B, k]]`
  - `[[B, k]]`
  - `[[E, d_model]]`
  - `[[T, 1, 1]]`
  - `[[T, 1, 2], [T, 1, 2]]`
  - `[[T, 1, E]]`
  - `[[T, 1], [T, 1]]`
  - `[[T, 1]]`
  - `[[T, E]]`
  - `[[T, d_model]]`
  - `[[T, k], [T, k]]`
  - `[[T, k]]`
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
  - `[[B, 1, 1, d_rope/2]]`
  - `[[B, 1, 1, n_h]]`
  - `[[B, 1, T, d_rope/2]]`
  - `[[B, 1, T, n_h]]`
  - `[[B, 1, c_kv], [B, 1, n_h]]`
  - `[[B, 1, d_rope/2]]`
  - `[[B, 1, n_h*d_v]]`
  - `[[B, 1, n_h, d_nope+d_rope]]`
  - `[[B, 1, n_h, d_nope+d_v]]`
  - `[[B, 1, n_h, d_nope]]`
  - `[[B, T, c_kv], [B, T, n_h]]`
  - `[[B, T, d_rope/2]]`
  - `[[B, T, n_h*d_v]]`
  - `[[B, T, n_h, d_nope+d_rope]]`
  - `[[B, T, n_h, d_nope+d_v]]`
  - `[[B, T, n_h, d_nope]]`
  - `[[B, n_h, 1, 2*d_nope]]`
  - `[[B, n_h, 1, T+1]]`
  - `[[B, n_h, 1, d_head]]`
  - `[[B, n_h, 1, d_nope+d_rope]]`
  - `[[B, n_h, 1, d_nope], [B, n_h, 1, d_head]]`
  - `[[B, n_h, 1, d_nope], [B, n_h, 1, d_nope]]`
  - `[[B, n_h, 1, d_nope]]`
  - `[[B, n_h, 1, d_rope/2]]`
  - `[[B, n_h, T+1, d_nope+d_rope]]`
  - `[[B, n_h, T+1, d_nope]]`
  - `[[B, n_h, T, 2*d_nope]]`
  - `[[B, n_h, T, T]]`
  - `[[B, n_h, T, d_head]]`
  - `[[B, n_h, T, d_nope+d_rope]]`
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
  - `[[B, (n_h+2*n_kv)*d_head]]`
  - `[[B, 1, (n_h+2*n_kv)*d_head]]`
  - `[[B, T, (n_h+2*n_kv)*d_head]]`
  - `[[B, c_q]]`
  - `[[T, (n_h+2*n_kv)*d_head]]`
  - `[[T, c_q]]`
  - `[[c_q, (n_h+2*n_kv)*d_head]]`
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
