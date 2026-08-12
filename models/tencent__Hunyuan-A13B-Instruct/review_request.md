# 검토 의뢰서 — tencent/Hunyuan-A13B-Instruct

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `hunyuan_v1_moe`
- 판단 필요: **1건**

## 증거 — 이미 받아둔 실제 소스

- `develop/sources/modeling_hunyuan_v1_moe.py` — 있음, 이 파일을 열어서 판정한다
- `develop/sources/configuration_hunyuan_v1_moe.py` — 있음, 이 파일을 열어서 판정한다

- 온라인 원본: https://github.com/huggingface/transformers/tree/main/src/transformers/models/hunyuan_v1_moe

그 밖의 재료: `full/review.md`(리뷰 패킷 — shape 별 실제 행 표본), `structure.yaml`(이 모델의 심볼 표), `full/<phase>.csv`(전체 operator 표).

## 판단이 필요한 것

### 1. 이 config 필드가 정말 이 뜻인가

값은 로드된 config 에 있지만 이 모델의 config 클래스가 선언한 필드가 아니다 (체크포인트 `config.json` 에서 온 값). 클래스가 뜻을 보증하지 않으므로 modeling 소스에서 이 필드가 실제로 어떻게 쓰이는지 확인해야 한다.

- `d_moe ← moe_intermediate_size`

## 기계적으로 이미 확인된 것 — 다시 묻지 말 것

- **심볼이 읽은 config 필드**: 1건이 클래스 선언 밖 — 위 1절 참고
- **정사각 축**: 소스에서 정사각 생성/reshape 과 대응이 확인된 축 없음
- **모듈이 읽는 config 속성**: `__init__` 에서 config 를 읽는 클래스 8개를 소스에서 확인했다. 그 목록이 각 모듈의 폭이 가질 수 있는 이름의 전부다.
- **가중치 축 ↔ 모듈 소속**: 가중치 축의 이름이 전부 그 모듈(또는 그 부모)이 실제로 읽는 config 필드에서 나왔다. 이 축들은 값이 아니라 소스로 확인된 것이다.

## 전수 점검 — 이 모델이 쓰는 이름 전부

위 절이 '풀리지 않은 것'이라면 여기는 **전부**다. 규칙이 자신 있게 붙인 이름도 틀릴 수 있고, 그런 건 미결 목록에 절대 오르지 않는다. 한 줄씩 읽고 **그 모듈에서 그 이름이 말이 되는지** 보라.

### A. 붙은 이름 전부 (17종)

| 라벨 | 값 | 나타나는 모듈 | 축 수 |
|---|---|---|---|
| `B` |  | `model.layers.*.self_attn`, `model.layers.*.input_layernorm`, `model.layers.*.self_attn.query_layernorm`, `model.layers.*.self_attn.key_layernorm` 외 52개 | 19294 |
| `T` |  | `model.layers.*.self_attn`, `model.layers.*.mlp.gate`, `model.layers.*.input_layernorm`, `model.layers.*.self_attn.query_layernorm` 외 52개 | 11612 |
| `d_head` | 128 | `model.layers.*.self_attn`, `model.layers.*.self_attn.query_layernorm`, `model.layers.*.self_attn.key_layernorm`, `model.rotary_emb` | 9306 |
| `d_model` | 4096 | `model.layers.*.mlp.experts`, `model.layers.*.input_layernorm`, `model.layers.*.post_attention_layernorm`, `model.layers.*.mlp.shared_mlp.gate_proj` 외 44개 | 9022 |
| `n_h` | 32 | `model.layers.*.self_attn`, `model.layers.*.self_attn.query_layernorm` | 6720 |
| `n_kv` | 8 | `model.layers.*.self_attn`, `model.layers.*.self_attn.key_layernorm` | 4736 |
| `n_h*d_head` |  | `model.layers.*.self_attn.o_proj`, `model.layers.*.self_attn.q_proj`, `model.layers.*.self_attn.k_proj`, `model.layers.*.self_attn.v_proj` 외 1개 | 3136 |
| `k` | 8 | `model.layers.*.mlp.experts`, `model.layers.*.mlp.gate`, `model.layers.*.mlp.experts.act_fn` | 2976 |
| `T+1` |  | `model.layers.*.self_attn`, `model` | 2223 |
| `k*T` |  | `model.layers.*.mlp.experts`, `model.layers.*.mlp.experts.act_fn` | 2208 |
| `E` | 64 | `model.layers.*.mlp.experts`, `model.layers.*.mlp.gate.wg`, `model.layers.*.mlp.gate` | 2112 |
| `d_moe` | 3072 | `model.layers.*.mlp.experts`, `model.layers.*.mlp.shared_mlp.gate_proj`, `model.layers.*.mlp.shared_mlp.up_proj`, `model.layers.*.mlp.shared_mlp.down_proj` 외 1개 | 2080 |
| `n_kv*d_head` |  | `model.layers.*.self_attn.k_proj`, `model.layers.*.self_attn.v_proj`, `model.layers.*.self_attn` | 1536 |
| `d_head/2` |  | `model.layers.*.self_attn`, `model.rotary_emb` | 828 |
| `2*d_moe` |  | `model.layers.*.mlp.experts` | 704 |
| `n_h/n_kv` |  | `model.layers.*.self_attn` | 512 |
| `V` | 128167 | `lm_head`, `model.embed_tokens` | 24 |

### B. 이름 없이 남은 정수 전부 (5쌍)

**여기가 필터가 못 보던 자리다.** 정수가 남는 것 자체는 정상이다(루프 인덱스, 피연산자 개수, 브로드캐스트 축). 문제는 **이름이 있어야 하는데 없는 경우**이고, 마지막 열이 그 신호다 — 이 모델의 심볼과 값이 같다면 스코프가 그 모듈을 못 덮고 있을 수 있다. 실제로 `n_hc`(=4)가 그렇게 정수로 남아 있었다.

| 모듈 | 정수 | 축 수 | 같은 값의 심볼 |
|---|---|---|---|
| `model.layers.*.mlp.shared_mlp.down_proj` | 3072 | 544 | `d_ff`, `d_moe` |
| `model.layers.*.mlp.shared_mlp.gate_proj` | 3072 | 192 | `d_ff`, `d_moe` |
| `model.layers.*.mlp.shared_mlp.up_proj` | 3072 | 192 | `d_ff`, `d_moe` |
| `model.layers.*.mlp.shared_mlp` | 3072 | 192 | `d_ff`, `d_moe` |
| `model.layers.*.mlp.shared_mlp.act_fn` | 3072 | 128 | `d_ff`, `d_moe` |

### C. 모듈이 내는 출력 shape 전부 (57개 모듈 / 260종)

모듈 하나가 어떤 모양을 내놓는지 전부 적었다. 어떤 모듈에 **있을 수 없는 이름**이 섞여 있는지 보는 자리다(예: attention head 수가 Mamba mixer 안에, 전문가 수가 self_attn 안에).

- `(root)`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[]`
- `lm_head`
  - `[[B, 1, V]]`
  - `[[B, T, V]]`
  - `[[B, V]]`
  - `[[B, d_model]]`
  - `[[T, V]]`
  - `[[T, d_model]]`
  - `[[d_model, V]]`
  - `[]`
- `model`
  - `[[B, 1, 1, 1]]`
  - `[[B, 1, 1, T+1]]`
  - `[[B, 1, 1, T]]`
  - `[[B, 1, 1]]`
  - `[[B, 1, T+1]]`
  - `[[B, 1, T, 1]]`
  - `[[B, 1, T, T]]`
  - `[[B, 1, T]]`
  - `[[B, 1]]`
  - `[[B, T+1]]`
  - `[[B, T]]`
  - `[[B]]`
  - `[[T+1]]`
  - `[[T]]`
  - `[[]]`
- `model.embed_tokens`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.*.input_layernorm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
- `model.layers.*.mlp`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[T, d_model]]`
  - `[]`
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
  - `[]`
- `model.layers.*.mlp.experts.act_fn`
  - `[[k*T, d_moe]]`
  - `[[k, d_moe]]`
- `model.layers.*.mlp.gate`
  - `[[B, 1]]`
  - `[[B, E]]`
  - `[[B, d_model]]`
  - `[[B, k], [B, k]]`
  - `[[B, k]]`
  - `[[T, 1]]`
  - `[[T, E]]`
  - `[[T, d_model]]`
  - `[[T, k], [T, k]]`
  - `[[T, k]]`
  - `[]`
- `model.layers.*.mlp.gate.wg`
  - `[[B, E]]`
  - `[[T, E]]`
  - `[[d_model, E]]`
  - `[]`
- `model.layers.*.mlp.shared_mlp`
  - `[[B, 1, 3072]]`
  - `[[B, T, 3072]]`
- `model.layers.*.mlp.shared_mlp.act_fn`
  - `[[B, 1, 3072]]`
  - `[[B, T, 3072]]`
- `model.layers.*.mlp.shared_mlp.down_proj`
  - `[[3072, d_model]]`
  - `[[B, 1, d_model]]`
  - `[[B, 3072]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[T, 3072]]`
  - `[[T, d_model]]`
  - `[]`
- `model.layers.*.mlp.shared_mlp.gate_proj`
  - `[[B, 1, 3072]]`
  - `[[B, 3072]]`
  - `[[B, T, 3072]]`
  - `[[B, d_model]]`
  - `[[T, 3072]]`
  - `[[T, d_model]]`
  - `[[d_model, d_moe]]`
  - `[]`
- `model.layers.*.mlp.shared_mlp.up_proj`
  - `[[B, 1, 3072]]`
  - `[[B, 3072]]`
  - `[[B, T, 3072]]`
  - `[[B, d_model]]`
  - `[[T, 3072]]`
  - `[[T, d_model]]`
  - `[[d_model, d_moe]]`
  - `[]`
- `model.layers.*.post_attention_layernorm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
- `model.layers.*.self_attn`
  - `[[0]]`
  - `[[B, 1, 1, d_head]]`
  - `[[B, 1, T, d_head]]`
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
  - `[[n_h, B, T+1]]`
  - `[[n_h, B, d_head]]`
  - `[[n_h, T+1, d_head]]`
  - `[[n_h, T, T]]`
  - `[[n_h, T, d_head]]`
  - `[[n_h, d_head, T+1]]`
  - `[[n_h, d_head, T]]`
  - `[]`
- `model.layers.*.self_attn.k_proj`
  - `[[B, 1, n_kv*d_head]]`
  - `[[B, T, n_kv*d_head]]`
  - `[[B, d_model]]`
  - `[[B, n_kv*d_head]]`
  - `[[T, d_model]]`
  - `[[T, n_kv*d_head]]`
  - `[[d_model, n_kv*d_head]]`
  - `[]`
- `model.layers.*.self_attn.key_layernorm`
  - `[[B, n_kv, 1, 1]]`
  - `[[B, n_kv, 1, d_head]]`
  - `[[B, n_kv, T, 1]]`
  - `[[B, n_kv, T, d_head]]`
- `model.layers.*.self_attn.o_proj`
  - `[[B, 1, n_h*d_head]]`
  - `[[B, T, n_h*d_head]]`
  - `[[B, n_h*d_head]]`
  - `[[T, n_h*d_head]]`
  - `[[n_h*d_head, n_h*d_head]]`
  - `[]`
- `model.layers.*.self_attn.q_proj`
  - `[[B, 1, n_h*d_head]]`
  - `[[B, T, n_h*d_head]]`
  - `[[B, d_model]]`
  - `[[B, n_h*d_head]]`
  - `[[T, d_model]]`
  - `[[T, n_h*d_head]]`
  - `[[d_model, n_h*d_head]]`
  - `[]`
- `model.layers.*.self_attn.query_layernorm`
  - `[[B, n_h, 1, 1]]`
  - `[[B, n_h, 1, d_head]]`
  - `[[B, n_h, T, 1]]`
  - `[[B, n_h, T, d_head]]`
- `model.layers.*.self_attn.v_proj`
  - `[[B, 1, n_kv*d_head]]`
  - `[[B, T, n_kv*d_head]]`
  - `[[B, d_model]]`
  - `[[B, n_kv*d_head]]`
  - `[[T, d_model]]`
  - `[[T, n_kv*d_head]]`
  - `[[d_model, n_kv*d_head]]`
  - `[]`
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
  - `[[]]`
  - `[]`

## 이 의뢰서를 처리하는 법

`review/prompt.md` 를 LLM 에 넘기고 이 모델을 지정한다. 판정 4종과 근거 요건, 결과를 어디에 어떤 형식으로 쓰는지는 전부 `review/` 안에 있다.
