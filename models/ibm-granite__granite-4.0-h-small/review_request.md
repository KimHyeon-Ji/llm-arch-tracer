# 검토 의뢰서 — ibm-granite/granite-4.0-h-small

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `granitemoehybrid`
- 판단 필요: **3건**

## 증거 — 이미 받아둔 실제 소스

- `develop/sources/modeling_granitemoehybrid.py` — 있음, 이 파일을 열어서 판정한다
- `develop/sources/configuration_granitemoehybrid.py` — 있음, 이 파일을 열어서 판정한다

- 온라인 원본: https://github.com/huggingface/transformers/tree/main/src/transformers/models/granitemoehybrid

그 밖의 재료: `full/review.md`(리뷰 패킷 — shape 별 실제 행 표본), `structure.yaml`(이 모델의 심볼 표), `full/<phase>.csv`(전체 operator 표).

## 판단이 필요한 것

### 3. 이름 붙일 근거가 없는 config 필드

모듈 폭으로 쓰이는데 심볼 표에 등록돼 있지 않다. 소스에서 무엇인지 확인하고 `rules/symbols.yaml` 에 등록하면 다음 실행부터 자동으로 잡힌다.

- `{'field': 'logits_scaling', 'value': 16, 'modules': 1}`
- `{'field': 'embedding_multiplier', 'value': 12, 'modules': 1}`

### 6. 값이 겹쳐 **임의로** 고른 축

두 심볼이 같은 값을 갖는 자리다. 규칙에는 고를 근거가 없고, 이긴 쪽은 전역 우선순위 — 즉 **관례**로 정해졌다. 이름이 맞을 수도 있지만 파이프라인은 그걸 알지 못한다. 표에서는 확신 있는 라벨과 똑같이 보인다.

**소스를 열어 어느 쪽인지 확정하는 것이 여기서 할 일이다.** 확정되면 `rules/label_overrides.yaml` 에 근거와 함께 못 박는다(review/05-overrides.md). 출신으로만 구별되는 경우라면 그렇게 적고 `open` 으로 남긴다.

- `d_state vs n_h_ssm` in `model.layers.*.mamba` — 값 128 를 두고 후보가 2개, 11556축

## 기계적으로 이미 확인된 것 — 다시 묻지 말 것

- **심볼이 읽은 config 필드**: 전부 이 모델의 config 클래스(또는 상속/프로퍼티/getattr 기본값)에 존재한다
- **정사각 축**: `d_chunk` ← 소스의 `chunk_size` ← `mamba_chunk_size`
- **모듈이 읽는 config 속성**: `__init__` 에서 config 를 읽는 클래스 10개를 소스에서 확인했다. 그 목록이 각 모듈의 폭이 가질 수 있는 이름의 전부다.
- **가중치 축 ↔ 모듈 소속**: 가중치 축의 이름이 전부 그 모듈(또는 그 부모)이 실제로 읽는 config 필드에서 나왔다. 이 축들은 값이 아니라 소스로 확인된 것이다.

## 전수 점검 — 이 모델이 쓰는 이름 전부

위 절이 '풀리지 않은 것'이라면 여기는 **전부**다. 규칙이 자신 있게 붙인 이름도 틀릴 수 있고, 그런 건 미결 목록에 절대 오르지 않는다. 한 줄씩 읽고 **그 모듈에서 그 이름이 말이 되는지** 보라.

### A. 붙은 이름 전부 (28종)

| 라벨 | 값 | 나타나는 모듈 | 축 수 |
|---|---|---|---|
| `B` |  | `model.layers.*.mamba`, `model.layers.*.mamba.norm`, `model.layers.*.input_layernorm`, `model.layers.*.post_attention_layernorm` 외 61개 | 21960 |
| `d_state` | 128 | `model.layers.*.mamba` | 11556 |
| `d_model` | 4096 | `model.layers.*.block_sparse_moe.experts`, `model.layers.*.input_layernorm`, `model.layers.*.post_attention_layernorm`, `model.layers.*.shared_mlp.input_linear` 외 53개 | 8518 |
| `T` |  | `model.layers.*.mamba`, `model.layers.*.mamba.norm`, `model.layers.*.input_layernorm`, `model.layers.*.post_attention_layernorm` 외 61개 | 7669 |
| `d_chunk` | 256 | `model.layers.*.mamba` | 5652 |
| `d_head_ssm` | 64 | `model.layers.*.mamba` | 4248 |
| `n_h_ssm` | 128 | `model.layers.*.mamba` | 3672 |
| `k` | 10 | `model.layers.*.block_sparse_moe.experts`, `model.layers.*.block_sparse_moe.router`, `model.layers.*.block_sparse_moe.experts.act_fn` | 3000 |
| `d_inner` |  | `model.layers.*.mamba.norm`, `model.layers.*.mamba.out_proj`, `model.layers.*.mamba` | 2232 |
| `k*T` |  | `model.layers.*.block_sparse_moe.experts`, `model.layers.*.block_sparse_moe.experts.act_fn` | 2200 |
| `E` | 72 | `model.layers.*.block_sparse_moe.experts`, `model.layers.*.block_sparse_moe.router` | 1920 |
| `d_inner+2*n_g*d_state` |  | `model.layers.*.mamba`, `model.layers.*.mamba.conv1d`, `model.layers.*.mamba.act` | 1800 |
| `d_shared` | 1536 | `model.layers.*.shared_mlp.output_linear`, `model.layers.*.shared_mlp`, `model.layers.*.shared_mlp.activation` | 1200 |
| `d_moe` | 768 | `model.layers.*.block_sparse_moe.experts`, `model.layers.*.block_sparse_moe.experts.act_fn` | 1040 |
| `d_conv` | 4 | `model.layers.*.mamba`, `model.layers.*.mamba.conv1d` | 792 |
| `2*d_shared` |  | `model.layers.*.shared_mlp.input_linear`, `model.layers.*.shared_mlp` | 720 |
| `2*d_inner+2*n_g*d_state+n_h_ssm` |  | `model.layers.*.mamba.in_proj`, `model.layers.*.mamba` | 648 |
| `2*d_moe` |  | `model.layers.*.block_sparse_moe.experts` | 560 |
| `d_head` | 128 | `model.layers.*.self_attn` | 432 |
| `n_h` | 32 | `model.layers.*.self_attn` | 376 |
| `T+1` |  | `model.layers.*.self_attn`, `model` | 211 |
| `n_h*d_head` |  | `model.layers.*.self_attn.o_proj`, `model.layers.*.self_attn.q_proj`, `model.layers.*.self_attn` | 208 |
| `n_kv` | 8 | `model.layers.*.self_attn` | 200 |
| `n_kv*d_head` |  | `model.layers.*.self_attn.k_proj`, `model.layers.*.self_attn.v_proj`, `model.layers.*.self_attn` | 144 |
| `d_conv+1` |  | `model.layers.*.mamba` | 108 |
| `T+d_conv-1` |  | `model.layers.*.mamba.conv1d`, `model.layers.*.mamba` | 72 |
| `n_h/n_kv` |  | `model.layers.*.self_attn` | 64 |
| `V` | 100352 | `lm_head`, `model.embed_tokens`, `(root)` | 24 |

### B. 이름 없이 남은 정수 전부 (1쌍)

**여기가 필터가 못 보던 자리다.** 정수가 남는 것 자체는 정상이다(루프 인덱스, 피연산자 개수, 브로드캐스트 축). 문제는 **이름이 있어야 하는데 없는 경우**이고, 마지막 열이 그 신호다 — 이 모델의 심볼과 값이 같다면 스코프가 그 모듈을 못 덮고 있을 수 있다. 실제로 `n_hc`(=4)가 그렇게 정수로 남아 있었다.

| 모듈 | 정수 | 축 수 | 같은 값의 심볼 |
|---|---|---|---|
| `model.layers.*.mamba` | 2 | 2556 | — |

### C. 모듈이 내는 출력 shape 전부 (66개 모듈 / 322종)

모듈 하나가 어떤 모양을 내놓는지 전부 적었다. 어떤 모듈에 **있을 수 없는 이름**이 섞여 있는지 보는 자리다(예: attention head 수가 Mamba mixer 안에, 전문가 수가 self_attn 안에).

- `(root)`
  - `[[B, 1, V]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, V]]`
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
  - `[[B, 1, 1, T+1]]`
  - `[[B, 1, 1, T]]`
  - `[[B, 1, 1]]`
  - `[[B, 1, T+1]]`
  - `[[B, 1, T, 1]]`
  - `[[B, 1, T, T]]`
  - `[[B, 1, T]]`
  - `[[B, 1, d_model]]`
  - `[[B, 1]]`
  - `[[B, T+1]]`
  - `[[B, T, d_model]]`
  - `[[B, T]]`
  - `[[B]]`
  - `[[T+1]]`
  - `[[T]]`
  - `[[]]`
- `model.embed_tokens`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.*.block_sparse_moe`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[T, d_model]]`
- `model.layers.*.block_sparse_moe.experts`
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
- `model.layers.*.block_sparse_moe.experts.act_fn`
  - `[[k*T, d_moe]]`
  - `[[k, d_moe]]`
- `model.layers.*.block_sparse_moe.router`
  - `[[B, E]]`
  - `[[B, k], [B, k]]`
  - `[[B, k]]`
  - `[[T, E]]`
  - `[[T, k], [T, k]]`
  - `[[T, k]]`
  - `[[d_model, E]]`
- `model.layers.*.input_layernorm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
- `model.layers.*.mamba`
  - `[[2, 2]]`
  - `[[B, 1, 1, d_chunk, d_state, d_head_ssm]]`
  - `[[B, 1, 1, d_chunk, d_state, n_h_ssm]]`
  - `[[B, 1, 1, d_state, d_head_ssm, n_h_ssm]]`
  - `[[B, 1, 1, d_state]]`
  - `[[B, 1, d_chunk, 1, d_state, n_h_ssm]]`
  - `[[B, 1, d_chunk, d_chunk, d_state, 1]]`
  - `[[B, 1, d_chunk, d_chunk, d_state, d_head_ssm]]`
  - `[[B, 1, d_chunk, d_chunk, d_state, n_h_ssm]]`
  - `[[B, 1, d_chunk, d_chunk, d_state]]`
  - `[[B, 1, d_chunk, d_state, 1, n_h_ssm]]`
  - `[[B, 1, d_chunk, d_state, 1]]`
  - `[[B, 1, d_chunk, d_state, d_head_ssm, 1]]`
  - `[[B, 1, d_chunk, d_state, d_head_ssm, n_h_ssm]]`
  - `[[B, 1, d_chunk, d_state, d_head_ssm]]`
  - `[[B, 1, d_chunk, d_state, n_h_ssm]]`
  - `[[B, 1, d_chunk, d_state]]`
  - `[[B, 1, d_inner], [B, 1, d_inner+2*n_g*d_state], [B, 1, d_state]]`
  - `[[B, 1, d_inner]]`
  - `[[B, 1, d_state, d_head_ssm, n_h_ssm]]`
  - `[[B, 1, d_state, n_h_ssm]]`
  - `[[B, 1, d_state]]`
  - `[[B, 2, 1, d_state, d_head_ssm, n_h_ssm]]`
  - `[[B, 2, 2, d_state, 1, 1]]`
  - `[[B, 2, 2, d_state, 1]]`
  - `[[B, 2, 2, d_state, d_head_ssm, n_h_ssm]]`
  - `[[B, 2, 2, d_state]]`
  - `[[B, 2, d_state, d_head_ssm, n_h_ssm]]`
  - `[[B, T, 1, 1, d_state]]`
  - `[[B, T, 1, d_state, n_h_ssm]]`
  - `[[B, T, 1, d_state]]`
  - `[[B, T, d_inner+2*n_g*d_state]]`
  - `[[B, T, d_inner], [B, T, d_inner+2*n_g*d_state], [B, T, d_state]]`
  - `[[B, T, d_inner], [B, T, d_state], [B, T, d_state]]`
  - `[[B, T, d_inner]]`
  - `[[B, T, d_state, 1]]`
  - `[[B, T, d_state, d_head_ssm]]`
  - `[[B, T, d_state, n_h_ssm]]`
  - `[[B, T, d_state]]`
  - `[[B, d_chunk, d_state, d_head_ssm]]`
  - `[[B, d_chunk, d_state, n_h_ssm]]`
  - `[[B, d_chunk, d_state]]`
  - `[[B, d_inner+2*n_g*d_state, 1]]`
  - `[[B, d_inner+2*n_g*d_state, T]]`
  - `[[B, d_inner+2*n_g*d_state, d_conv+1]]`
  - `[[B, d_inner+2*n_g*d_state, d_conv]]`
  - `[[B, d_inner+2*n_g*d_state]]`
  - `[[B, d_inner], [B, d_state], [B, d_state]]`
  - `[[B, d_inner]]`
  - `[[B, d_state, 1, 1]]`
  - `[[B, d_state, 1, d_chunk, 1]]`
  - `[[B, d_state, 1, d_chunk, d_chunk]]`
  - `[[B, d_state, 1, d_chunk]]`
  - `[[B, d_state, 1, n_h_ssm]]`
  - `[[B, d_state, 1]]`
  - `[[B, d_state, 2, 1]]`
  - `[[B, d_state, 2, 2]]`
  - `[[B, d_state, 2]]`
  - `[[B, d_state, d_head_ssm, 1]]`
  - `[[B, d_state, d_head_ssm, n_h_ssm]]`
  - `[[B, d_state, d_head_ssm]]`
  - `[[B, d_state, n_h_ssm]]`
  - `[[B, d_state]]`
  - `[[d_chunk, d_chunk]]`
  - `[[d_inner+2*n_g*d_state, d_conv]]`
  - `[[d_state, B, 1]]`
  - `[[d_state, B]]`
  - `[[d_state, d_head_ssm, B]]`
  - `[[d_state, d_head_ssm, n_h_ssm]]`
  - `[[d_state, d_head_ssm]]`
  - `[[d_state, n_h_ssm, B]]`
  - `[[d_state]]`
- `model.layers.*.mamba.act`
  - `[[B, d_inner+2*n_g*d_state, T]]`
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
  - `[[B, 1, 1]]`
  - `[[B, 1, d_inner]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_inner]]`
- `model.layers.*.mamba.out_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_inner]]`
  - `[[B, d_model]]`
  - `[[T, d_inner]]`
  - `[[T, d_model]]`
  - `[[d_inner, d_model]]`
- `model.layers.*.post_attention_layernorm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
- `model.layers.*.self_attn`
  - `[[B, 1, n_h*d_head]]`
  - `[[B, 1, n_h, d_head]]`
  - `[[B, 1, n_kv, d_head]]`
  - `[[B, T, n_h*d_head]]`
  - `[[B, T, n_h, d_head]]`
  - `[[B, T, n_kv, d_head]]`
  - `[[B, n_h, 1, T+1]]`
  - `[[B, n_h, 1, d_head]]`
  - `[[B, n_h, T+1, d_head]]`
  - `[[B, n_h, T, T]]`
  - `[[B, n_h, T, d_head]]`
  - `[[B, n_h, d_head, T+1]]`
  - `[[B, n_h, d_head, T]]`
  - `[[B, n_kv, 1, T+1, d_head]]`
  - `[[B, n_kv, 1, T, d_head]]`
  - `[[B, n_kv, 1, d_head]]`
  - `[[B, n_kv, T+1, d_head]]`
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
  - `[[d_model, n_h*d_head]]`
- `model.layers.*.self_attn.v_proj`
  - `[[B, 1, n_kv*d_head]]`
  - `[[B, T, n_kv*d_head]]`
  - `[[B, d_model]]`
  - `[[B, n_kv*d_head]]`
  - `[[T, d_model]]`
  - `[[T, n_kv*d_head]]`
  - `[[d_model, n_kv*d_head]]`
- `model.layers.*.shared_mlp`
  - `[[B, 1, d_shared], [B, 1, d_shared]]`
  - `[[B, 1, d_shared]]`
  - `[[B, T, d_shared], [B, T, d_shared]]`
  - `[[B, T, d_shared]]`
- `model.layers.*.shared_mlp.activation`
  - `[[B, 1, d_shared]]`
  - `[[B, T, d_shared]]`
- `model.layers.*.shared_mlp.input_linear`
  - `[[B, 1, 2*d_shared]]`
  - `[[B, 2*d_shared]]`
  - `[[B, T, 2*d_shared]]`
  - `[[B, d_model]]`
  - `[[T, 2*d_shared]]`
  - `[[T, d_model]]`
  - `[[d_model, 2*d_shared]]`
- `model.layers.*.shared_mlp.output_linear`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[B, d_shared]]`
  - `[[T, d_model]]`
  - `[[T, d_shared]]`
  - `[[d_shared, d_model]]`
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

## 이 의뢰서를 처리하는 법

`review/prompt.md` 를 LLM 에 넘기고 이 모델을 지정한다. 판정 4종과 근거 요건, 결과를 어디에 어떤 형식으로 쓰는지는 전부 `review/` 안에 있다.
