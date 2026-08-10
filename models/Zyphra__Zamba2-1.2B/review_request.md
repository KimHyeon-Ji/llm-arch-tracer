# 검토 의뢰서 — Zyphra/Zamba2-1.2B

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `zamba2`
- 판단 필요: **3건**

## 증거 — 이미 받아둔 실제 소스

- `develop/sources/modeling_zamba2.py` — 있음, 이 파일을 열어서 판정한다
- `develop/sources/configuration_zamba2.py` — 있음, 이 파일을 열어서 판정한다

- 온라인 원본: https://github.com/huggingface/transformers/tree/main/src/transformers/models/zamba2

그 밖의 재료: `full/review.md`(리뷰 패킷 — shape 별 실제 행 표본), `structure.yaml`(이 모델의 심볼 표), `full/<phase>.csv`(전체 operator 표).

## 판단이 필요한 것

### 2. 이 정사각 축이 정말 같은 이름 두 번인가

`[..., X, X]` 로 렌더됐는데, 그 이름이 읽은 config 필드에서 나온 정사각 reshape 을 modeling 소스에서 찾지 못했다. 두 축 크기가 우연히 같은 것일 수 있다.

- `d_attn`

### 4. 규칙 없이 산술로 지은 이름

값이 맞아떨어져서 붙인 이름이다. 산술적으로 참이어도 틀린 이름일 수 있으므로 (예: RoPE 절반 차원) 소스에서 확인이 필요하다.

- `2*d_ff` in `model.layers.*.shared_transformer.feed_forward.gate_up_proj (레이어 6개)` — heur_multiple, 126축
- `2*d_ff` in `model.layers.*.shared_transformer.feed_forward.gate_up_proj_adapter_list.*.1 (레이어 6개)` — heur_multiple, 144축

## 기계적으로 이미 확인된 것 — 다시 묻지 말 것

- **심볼이 읽은 config 필드**: 전부 이 모델의 config 클래스(또는 상속/프로퍼티/getattr 기본값)에 존재한다
- **정사각 축**: `d_chunk` ← 소스의 `chunk_size` ← `chunk_size`
- **모듈이 읽는 config 속성**: `__init__` 에서 config 를 읽는 클래스 9개를 소스에서 확인했다. 그 목록이 각 모듈의 폭이 가질 수 있는 이름의 전부다.

## 전수 점검 — 이 모델이 쓰는 이름 전부

위 절이 '풀리지 않은 것'이라면 여기는 **전부**다. 규칙이 자신 있게 붙인 이름도 틀릴 수 있고, 그런 건 미결 목록에 절대 오르지 않는다. 한 줄씩 읽고 **그 모듈에서 그 이름이 말이 되는지** 보라.

### A. 붙은 이름 전부 (23종)

| 라벨 | 값 | 나타나는 모듈 | 축 수 |
|---|---|---|---|
| `B` |  | `model.layers.*.mamba`, `model.layers.*.mamba_decoder.mamba`, `model.layers.*.mamba.norm`, `model.layers.*.shared_transformer.self_attn` 외 70개 | 19742 |
| `d_head_ssm` | 64 | `model.layers.*.mamba`, `model.layers.*.mamba_decoder.mamba` | 11210 |
| `d_chunk` | 256 | `model.layers.*.mamba`, `model.layers.*.mamba_decoder.mamba` | 6118 |
| `T` |  | `model.layers.*.mamba`, `model.layers.*.shared_transformer.self_attn`, `model.layers.*.mamba.norm`, `model.layers.*.input_layernorm` 외 70개 | 5902 |
| `d_state` | 128 | `model.layers.*.mamba`, `model.layers.*.mamba_decoder.mamba` | 5092 |
| `n_h_ssm` | 64 | `model.layers.*.mamba`, `model.layers.*.mamba_decoder.mamba` | 4712 |
| `d_model` | 2048 | `model.layers.*.input_layernorm`, `model.layers.*.mamba.in_proj`, `model.layers.*.mamba.out_proj`, `model.layers.*.linear` 외 47개 | 2830 |
| `d_inner` |  | `model.layers.*.mamba.norm`, `model.layers.*.mamba.out_proj`, `model.layers.*.mamba`, `model.layers.*.mamba_decoder.mamba.norm` 외 2개 | 2280 |
| `d_inner+2*n_g*d_state` |  | `model.layers.*.mamba`, `model.layers.*.mamba_decoder.mamba`, `model.layers.*.mamba.conv1d`, `model.layers.*.mamba.act` 외 2개 | 2052 |
| `d_head` | 128 | `model.layers.*.shared_transformer.self_attn`, `model.layers.*.shared_transformer.self_attn.linear_q_adapter_list.*.0`, `model.layers.*.shared_transformer.self_attn.linear_q_adapter_list.*.1`, `model.layers.*.shared_transformer.self_attn.linear_k_adapter_list.*.0` 외 4개 | 1434 |
| `d_attn` | 4096 | `model.layers.*.shared_transformer.self_attn`, `model.layers.*.shared_transformer.self_attn.q_proj`, `model.layers.*.shared_transformer.self_attn.k_proj`, `model.layers.*.shared_transformer.self_attn.v_proj` 외 9개 | 1368 |
| `n_h` | 32 | `model.layers.*.shared_transformer.self_attn` | 1020 |
| `d_conv` | 4 | `model.layers.*.mamba`, `model.layers.*.mamba_decoder.mamba`, `model.layers.*.mamba.conv1d`, `model.layers.*.mamba_decoder.mamba.conv1d` | 912 |
| `2*d_inner+2*n_g*d_state+n_h_ssm` |  | `model.layers.*.mamba.in_proj`, `model.layers.*.mamba_decoder.mamba.in_proj`, `model.layers.*.mamba`, `model.layers.*.mamba_decoder.mamba` | 684 |
| `2*d_ff` |  | `model.layers.*.shared_transformer.feed_forward.gate_up_proj`, `model.layers.*.shared_transformer.feed_forward.gate_up_proj_adapter_list.*.1`, `model.layers.*.shared_transformer.feed_forward` | 240 |
| `r_lora` | 128 | `model.layers.*.shared_transformer.feed_forward.gate_up_proj_adapter_list.*.0`, `model.layers.*.shared_transformer.feed_forward.gate_up_proj_adapter_list.*.1` | 192 |
| `d_head/2` |  | `model.layers.*.shared_transformer.self_attn`, `model.rotary_emb` | 180 |
| `d_ff` | 8192 | `model.layers.*.shared_transformer.feed_forward.down_proj`, `model.layers.*.shared_transformer.feed_forward`, `model.layers.*.shared_transformer.feed_forward.act_fn` | 180 |
| `T+1` |  | `model.layers.*.shared_transformer.self_attn` | 156 |
| `n_h*d_head` |  | `model.layers.*.shared_transformer.self_attn.q_proj`, `model.layers.*.shared_transformer.self_attn.k_proj`, `model.layers.*.shared_transformer.self_attn.v_proj` | 144 |
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
  - `[[d_attn, n_h*d_head]]`
- `model.layers.*.shared_transformer.self_attn.linear_k_adapter_list.*.0`
  - `[[B, 1, d_head]]`
  - `[[B, T, d_head]]`
  - `[[B, d_attn]]`
  - `[[B, d_head]]`
  - `[[T, d_attn]]`
  - `[[T, d_head]]`
  - `[[d_attn, d_head]]`
- `model.layers.*.shared_transformer.self_attn.linear_k_adapter_list.*.1`
  - `[[B, 1, d_attn]]`
  - `[[B, T, d_attn]]`
  - `[[B, d_attn]]`
  - `[[B, d_head]]`
  - `[[T, d_attn]]`
  - `[[T, d_head]]`
  - `[[d_head, d_attn]]`
- `model.layers.*.shared_transformer.self_attn.linear_q_adapter_list.*.0`
  - `[[B, 1, d_head]]`
  - `[[B, T, d_head]]`
  - `[[B, d_attn]]`
  - `[[B, d_head]]`
  - `[[T, d_attn]]`
  - `[[T, d_head]]`
  - `[[d_attn, d_head]]`
- `model.layers.*.shared_transformer.self_attn.linear_q_adapter_list.*.1`
  - `[[B, 1, d_attn]]`
  - `[[B, T, d_attn]]`
  - `[[B, d_attn]]`
  - `[[B, d_head]]`
  - `[[T, d_attn]]`
  - `[[T, d_head]]`
  - `[[d_head, d_attn]]`
- `model.layers.*.shared_transformer.self_attn.linear_v_adapter_list.*.0`
  - `[[B, 1, d_head]]`
  - `[[B, T, d_head]]`
  - `[[B, d_attn]]`
  - `[[B, d_head]]`
  - `[[T, d_attn]]`
  - `[[T, d_head]]`
  - `[[d_attn, d_head]]`
- `model.layers.*.shared_transformer.self_attn.linear_v_adapter_list.*.1`
  - `[[B, 1, d_attn]]`
  - `[[B, T, d_attn]]`
  - `[[B, d_attn]]`
  - `[[B, d_head]]`
  - `[[T, d_attn]]`
  - `[[T, d_head]]`
  - `[[d_head, d_attn]]`
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
  - `[[d_attn, n_h*d_head]]`
- `model.layers.*.shared_transformer.self_attn.v_proj`
  - `[[B, 1, d_attn]]`
  - `[[B, T, d_attn]]`
  - `[[B, d_attn]]`
  - `[[T, d_attn]]`
  - `[[d_attn, n_h*d_head]]`
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
