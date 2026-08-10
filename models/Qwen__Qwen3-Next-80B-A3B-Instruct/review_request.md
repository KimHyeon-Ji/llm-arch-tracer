# 검토 의뢰서 — Qwen/Qwen3-Next-80B-A3B-Instruct

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `qwen3_next`
- 판단 필요: **5건**

## 증거 — 이미 받아둔 실제 소스

- `develop/sources/modeling_qwen3_next.py` — 있음, 이 파일을 열어서 판정한다
- `develop/sources/configuration_qwen3_next.py` — 있음, 이 파일을 열어서 판정한다

- 온라인 원본: https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_next

그 밖의 재료: `full/review.md`(리뷰 패킷 — shape 별 실제 행 표본), `structure.yaml`(이 모델의 심볼 표), `full/<phase>.csv`(전체 operator 표).

## 판단이 필요한 것

### 4. 규칙 없이 산술로 지은 이름

값이 맞아떨어져서 붙인 이름이다. 산술적으로 참이어도 틀린 이름일 수 있으므로 (예: RoPE 절반 차원) 소스에서 확인이 필요하다.

- `3*n_kv` in `model.layers.*.linear_attn (레이어 3개)` — heur_multiple, 90축
- `3*d_conv_lin` in `model.layers.*.linear_attn (레이어 3개)` — heur_multiple, 90축
- `n_h_lin_v+1` in `model.layers.*.linear_attn (레이어 2개)` — heur_plus1, 60축
- `n_kv*T` in `model.layers.*.linear_attn (레이어 2개)` — heur_product, 60축
- `3*n_h` in `model.layers.*.linear_attn (레이어 2개)` — heur_multiple, 60축

## 기계적으로 이미 확인된 것 — 다시 묻지 말 것

- **심볼이 읽은 config 필드**: 전부 이 모델의 config 클래스(또는 상속/프로퍼티/getattr 기본값)에 존재한다
- **정사각 축**: 소스에서 정사각 생성/reshape 과 대응이 확인된 축 없음
- **모듈이 읽는 config 속성**: `__init__` 에서 config 를 읽는 클래스 10개를 소스에서 확인했다. 그 목록이 각 모듈의 폭이 가질 수 있는 이름의 전부다.

## 전수 점검 — 이 모델이 쓰는 이름 전부

위 절이 '풀리지 않은 것'이라면 여기는 **전부**다. 규칙이 자신 있게 붙인 이름도 틀릴 수 있고, 그런 건 미결 목록에 절대 오르지 않는다. 한 줄씩 읽고 **그 모듈에서 그 이름이 말이 되는지** 보라.

### A. 붙은 이름 전부 (34종)

| 라벨 | 값 | 나타나는 모듈 | 축 수 |
|---|---|---|---|
| `B` |  | `model.layers.*.linear_attn`, `model.layers.*.self_attn`, `model.layers.*.input_layernorm`, `model.layers.*.post_attention_layernorm` 외 73개 | 99644 |
| `n_h_lin_v` | 32 | `model.layers.*.linear_attn`, `model.layers.*.linear_attn.norm` | 84096 |
| `d_rope` |  | `model.layers.*.linear_attn`, `model.layers.*.self_attn`, `model.layers.*.linear_attn.in_proj_ba`, `model.rotary_emb` | 33842 |
| `d_head_lin_k` | 128 | `model.layers.*.linear_attn`, `model.layers.*.linear_attn.norm` | 13716 |
| `T` |  | `model.layers.*.linear_attn`, `model.layers.*.self_attn`, `model.layers.*.mlp.gate`, `model.layers.*.input_layernorm` 외 72개 | 13700 |
| `d_model` | 2048 | `model.layers.*.mlp.experts`, `model.layers.*.input_layernorm`, `model.layers.*.post_attention_layernorm`, `model.layers.*.mlp` 외 65개 | 13210 |
| `n_h` | 16 | `model.layers.*.linear_attn`, `model.layers.*.self_attn`, `model.layers.*.self_attn.q_norm` | 5772 |
| `d_moe` | 512 | `model.layers.*.mlp.experts`, `model.layers.*.mlp.shared_expert.gate_proj`, `model.layers.*.mlp.shared_expert.up_proj`, `model.layers.*.mlp.shared_expert.down_proj` 외 3개 | 4416 |
| `k` | 10 | `model.layers.*.mlp.experts`, `model.layers.*.mlp.gate`, `model.layers.*.linear_attn`, `model.layers.*.mlp.experts.act_fn` | 3948 |
| `d_head` | 256 | `model.layers.*.self_attn`, `model.layers.*.linear_attn`, `model.layers.*.self_attn.q_norm`, `model.layers.*.self_attn.k_norm` | 3216 |
| `k*T` |  | `model.layers.*.mlp.experts`, `model.layers.*.mlp.experts.act_fn` | 2640 |
| `n_kv` | 2 | `model.layers.*.self_attn`, `model.layers.*.self_attn.k_norm`, `model.layers.*.linear_attn` | 1980 |
| `2*n_h*d_head` |  | `model.layers.*.linear_attn`, `model.layers.*.self_attn.q_proj`, `model.layers.*.linear_attn.conv1d`, `model.layers.*.self_attn` | 1944 |
| `E` | 512 | `model.layers.*.mlp.gate`, `model.layers.*.mlp.experts` | 1536 |
| `d_head_lin_v` | 128 | `model.layers.*.linear_attn` | 1404 |
| `n_h*d_head` |  | `model.layers.*.linear_attn.out_proj`, `model.layers.*.linear_attn`, `model.layers.*.self_attn.o_proj`, `model.layers.*.self_attn` | 1296 |
| `d_conv_lin` | 4 | `model.layers.*.linear_attn`, `model.layers.*.linear_attn.conv1d` | 1116 |
| `n_h_lin_v*T` |  | `model.layers.*.linear_attn.norm`, `model.layers.*.linear_attn` | 1044 |
| `T+1` |  | `model.layers.*.self_attn`, `model.layers.*.linear_attn` | 828 |
| `2*d_moe` |  | `model.layers.*.mlp.experts` | 672 |
| `2*n_k*d_k+2*n_v*d_v` |  | `model.layers.*.linear_attn.in_proj_qkvz`, `model.layers.*.linear_attn` | 648 |
| `d_rope/2` |  | `model.layers.*.self_attn`, `model.layers.*.linear_attn`, `model.rotary_emb` | 576 |
| `n_h/n_kv` |  | `model.layers.*.linear_attn`, `model.layers.*.self_attn` | 444 |
| `n_kv*d_head` |  | `model.layers.*.self_attn.k_proj`, `model.layers.*.self_attn.v_proj`, `model.layers.*.self_attn` | 432 |
| `n_h+2*n_kv` |  | `model.layers.*.linear_attn`, `model.layers.*.linear_attn.conv1d` | 324 |
| `3*n_kv` |  | `model.layers.*.linear_attn` | 252 |
| `3*d_conv_lin` |  | `model.layers.*.linear_attn` | 252 |
| `n_h_lin_v+1` |  | `model.layers.*.linear_attn` | 252 |
| `n_kv*T` |  | `model.layers.*.linear_attn` | 252 |
| `3*n_h` |  | `model.layers.*.linear_attn` | 252 |
| `3*d_head` |  | `model.layers.*.linear_attn` | 144 |
| `d_head-d_rope` |  | `model.layers.*.self_attn` | 96 |
| `2*d_head` |  | `model.layers.*.self_attn` | 48 |
| `V` | 151936 | `lm_head`, `model.embed_tokens` | 20 |

### B. 이름 없이 남은 정수 전부 (63쌍)

**여기가 필터가 못 보던 자리다.** 정수가 남는 것 자체는 정상이다(루프 인덱스, 피연산자 개수, 브로드캐스트 축). 문제는 **이름이 있어야 하는데 없는 경우**이고, 마지막 열이 그 신호다 — 이 모델의 심볼과 값이 같다면 스코프가 그 모듈을 못 덮고 있을 수 있다. 실제로 `n_hc`(=4)가 그렇게 정수로 남아 있었다.

| 모듈 | 정수 | 축 수 | 같은 값의 심볼 |
|---|---|---|---|
| `model.layers.*.linear_attn` | 64 | 4608 | — |
| `model.layers.*.linear_attn` | 2 | 1908 | `n_kv` |
| `model.layers.*.linear_attn` | 5 | 1116 | — |
| `model.layers.*.linear_attn` | 3 | 1008 | — |
| `model.layers.*.linear_attn` | 7 | 1008 | — |
| `model.layers.*.linear_attn` | 9 | 1008 | — |
| `model.layers.*.linear_attn` | 11 | 1008 | — |
| `model.layers.*.linear_attn` | 13 | 1008 | — |
| `model.layers.*.linear_attn` | 14 | 1008 | — |
| `model.layers.*.linear_attn` | 15 | 1008 | — |
| `model.layers.*.linear_attn` | 19 | 1008 | — |
| `model.layers.*.linear_attn` | 21 | 1008 | — |
| `model.layers.*.linear_attn` | 22 | 1008 | — |
| `model.layers.*.linear_attn` | 23 | 1008 | — |
| `model.layers.*.linear_attn` | 24 | 1008 | — |
| `model.layers.*.linear_attn` | 25 | 1008 | — |
| `model.layers.*.linear_attn` | 26 | 1008 | — |
| `model.layers.*.linear_attn` | 27 | 1008 | — |
| `model.layers.*.linear_attn` | 28 | 1008 | — |
| `model.layers.*.linear_attn` | 29 | 1008 | — |
| `model.layers.*.linear_attn` | 30 | 1008 | — |
| `model.layers.*.linear_attn` | 31 | 1008 | — |
| `model.layers.*.linear_attn` | 35 | 1008 | — |
| `model.layers.*.linear_attn` | 36 | 1008 | — |
| `model.layers.*.linear_attn` | 37 | 1008 | — |
| `model.layers.*.linear_attn` | 38 | 1008 | — |
| `model.layers.*.linear_attn` | 39 | 1008 | — |
| `model.layers.*.linear_attn` | 40 | 1008 | — |
| `model.layers.*.linear_attn` | 41 | 1008 | — |
| `model.layers.*.linear_attn` | 42 | 1008 | — |
| `model.layers.*.linear_attn` | 43 | 1008 | — |
| `model.layers.*.linear_attn` | 44 | 1008 | — |
| `model.layers.*.linear_attn` | 45 | 1008 | — |
| `model.layers.*.linear_attn` | 46 | 1008 | — |
| `model.layers.*.linear_attn` | 47 | 1008 | — |
| `model.layers.*.linear_attn` | 49 | 1008 | — |
| `model.layers.*.linear_attn` | 50 | 1008 | — |
| `model.layers.*.linear_attn` | 51 | 1008 | — |
| `model.layers.*.linear_attn` | 52 | 1008 | — |
| `model.layers.*.linear_attn` | 53 | 1008 | — |
| `model.layers.*.linear_attn` | 54 | 1008 | — |
| `model.layers.*.linear_attn` | 55 | 1008 | — |
| `model.layers.*.linear_attn` | 56 | 1008 | — |
| `model.layers.*.linear_attn` | 57 | 1008 | — |
| `model.layers.*.linear_attn` | 58 | 1008 | — |
| `model.layers.*.linear_attn` | 59 | 1008 | — |
| `model.layers.*.linear_attn` | 60 | 1008 | — |
| `model.layers.*.linear_attn` | 61 | 1008 | — |
| `model.layers.*.linear_attn` | 62 | 1008 | — |
| `model.layers.*.linear_attn` | 63 | 1008 | — |
| `model.layers.*.linear_attn` | 4 | 756 | `d_conv_lin` |
| `model.layers.*.linear_attn` | 6 | 756 | — |
| `model.layers.*.linear_attn` | 8 | 756 | — |
| `model.layers.*.linear_attn` | 10 | 756 | `k` |
| `model.layers.*.linear_attn` | 12 | 756 | — |
| `model.layers.*.linear_attn` | 16 | 756 | `n_h`, `n_h_lin_k` |
| `model.layers.*.linear_attn` | 17 | 756 | — |
| `model.layers.*.linear_attn` | 18 | 756 | — |
| `model.layers.*.linear_attn` | 20 | 756 | — |
| `model.layers.*.linear_attn` | 32 | 756 | `n_h_lin_v` |
| `model.layers.*.linear_attn` | 33 | 756 | — |
| `model.layers.*.linear_attn` | 34 | 756 | — |
| `model.layers.*.linear_attn` | 48 | 756 | `L` |

### C. 모듈이 내는 출력 shape 전부 (78개 모듈 / 630종)

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
  - `[[d_model]]`
- `model.layers.*.linear_attn`
  - `[[2*n_h*d_head, B, d_conv_lin]]`
  - `[[2*n_h*d_head, d_conv_lin]]`
  - `[[B, 1, 2*n_h*d_head]]`
  - `[[B, 1, d_model], [B, 1, d_model], [B, 1, n_h*d_head]]`
  - `[[B, 1, d_model]]`
  - `[[B, 1, n_h*d_head]]`
  - `[[B, 1, n_h, 1, d_head_lin_k]]`
  - `[[B, 1, n_h, 2, d_head_lin_k]]`
  - `[[B, 1, n_h, 2], [B, 1, n_h, 2]]`
  - `[[B, 1, n_h, 2]]`
  - `[[B, 1, n_h, 3*d_head]]`
  - `[[B, 1, n_h, d_conv_lin]]`
  - `[[B, 1, n_h, d_head]]`
  - `[[B, 1, n_h, d_head_lin_k], [B, 1, n_h, d_head_lin_k], [B, 1, n_h, d_head], [B, 1, n_h, d_head]]`
  - `[[B, 1, n_h, d_head_lin_k]]`
  - `[[B, 1, n_h_lin_v, 1]]`
  - `[[B, 1, n_h_lin_v, d_head_lin_k]]`
  - `[[B, 1, n_h_lin_v]]`
  - `[[B, 2*n_h*d_head, 1]]`
  - `[[B, 2*n_h*d_head, 5]]`
  - `[[B, 2*n_h*d_head, T]]`
  - `[[B, 2*n_h*d_head, d_conv_lin]]`
  - `[[B, 2*n_h*d_head, n_kv]]`
  - `[[B, T, 2*n_h*d_head]]`
  - `[[B, T, d_model], [B, T, d_model], [B, T, n_h*d_head]]`
  - `[[B, T, d_model]]`
  - `[[B, T, n_h*d_head]]`
  - `[[B, T, n_h, 1, d_head_lin_k]]`
  - `[[B, T, n_h, 2, d_head_lin_k]]`
  - `[[B, T, n_h, 2], [B, T, n_h, 2]]`
  - `[[B, T, n_h, 2]]`
  - `[[B, T, n_h, 3*d_head]]`
  - `[[B, T, n_h, d_conv_lin]]`
  - `[[B, T, n_h, d_head]]`
  - `[[B, T, n_h, d_head_lin_k], [B, T, n_h, d_head_lin_k], [B, T, n_h, d_head], [B, T, n_h, d_head]]`
  - `[[B, T, n_h, d_head_lin_k]]`
  - `[[B, T, n_h_lin_v, 1]]`
  - `[[B, T, n_h_lin_v, d_head_lin_k]]`
  - `[[B, T, n_h_lin_v]]`
  - `[[B, n_h_lin_v, 1, 1, 1]]`
  - `[[B, n_h_lin_v, 1, 1, 64]]`
  - `[[B, n_h_lin_v, 1, 1, d_rope]]`
  - `[[B, n_h_lin_v, 1, 10, 10]]`
  - `[[B, n_h_lin_v, 1, 10, 1]]`
  - `[[B, n_h_lin_v, 1, 10, 64]]`
  - `[[B, n_h_lin_v, 1, 10]]`
  - `[[B, n_h_lin_v, 1, 11, 11]]`
  - `[[B, n_h_lin_v, 1, 11, 1]]`
  - `[[B, n_h_lin_v, 1, 11, 64]]`
  - `[[B, n_h_lin_v, 1, 11]]`
  - `[[B, n_h_lin_v, 1, 12, 12]]`
  - `[[B, n_h_lin_v, 1, 12, 1]]`
  - `[[B, n_h_lin_v, 1, 12, 64]]`
  - `[[B, n_h_lin_v, 1, 12]]`
  - `[[B, n_h_lin_v, 1, 13, 13]]`
  - `[[B, n_h_lin_v, 1, 13, 1]]`
  - `[[B, n_h_lin_v, 1, 13, 64]]`
  - `[[B, n_h_lin_v, 1, 13]]`
  - `[[B, n_h_lin_v, 1, 14, 14]]`
  - `[[B, n_h_lin_v, 1, 14, 1]]`
  - `[[B, n_h_lin_v, 1, 14, 64]]`
  - `[[B, n_h_lin_v, 1, 14]]`
  - `[[B, n_h_lin_v, 1, 15, 15]]`
  - `[[B, n_h_lin_v, 1, 15, 1]]`
  - `[[B, n_h_lin_v, 1, 15, 64]]`
  - `[[B, n_h_lin_v, 1, 15]]`
  - `[[B, n_h_lin_v, 1, 16, 16]]`
  - `[[B, n_h_lin_v, 1, 16, 1]]`
  - `[[B, n_h_lin_v, 1, 16, 64]]`
  - `[[B, n_h_lin_v, 1, 16]]`
  - `[[B, n_h_lin_v, 1, 17, 17]]`
  - `[[B, n_h_lin_v, 1, 17, 1]]`
  - `[[B, n_h_lin_v, 1, 17, 64]]`
  - `[[B, n_h_lin_v, 1, 17]]`
  - `[[B, n_h_lin_v, 1, 18, 18]]`
  - `[[B, n_h_lin_v, 1, 18, 1]]`
  - `[[B, n_h_lin_v, 1, 18, 64]]`
  - `[[B, n_h_lin_v, 1, 18]]`
  - `[[B, n_h_lin_v, 1, 19, 19]]`
  - `[[B, n_h_lin_v, 1, 19, 1]]`
  - `[[B, n_h_lin_v, 1, 19, 64]]`
  - `[[B, n_h_lin_v, 1, 19]]`
  - `[[B, n_h_lin_v, 1, 1]]`
  - `[[B, n_h_lin_v, 1, 2, 1]]`
  - `[[B, n_h_lin_v, 1, 2, 2]]`
  - `[[B, n_h_lin_v, 1, 2, 64]]`
  - `[[B, n_h_lin_v, 1, 20, 1]]`
  - `[[B, n_h_lin_v, 1, 20, 20]]`
  - `[[B, n_h_lin_v, 1, 20, 64]]`
  - `[[B, n_h_lin_v, 1, 20]]`
  - `[[B, n_h_lin_v, 1, 21, 1]]`
  - `[[B, n_h_lin_v, 1, 21, 21]]`
  - `[[B, n_h_lin_v, 1, 21, 64]]`
  - `[[B, n_h_lin_v, 1, 21]]`
  - `[[B, n_h_lin_v, 1, 22, 1]]`
  - `[[B, n_h_lin_v, 1, 22, 22]]`
  - `[[B, n_h_lin_v, 1, 22, 64]]`
  - `[[B, n_h_lin_v, 1, 22]]`
  - `[[B, n_h_lin_v, 1, 23, 1]]`
  - `[[B, n_h_lin_v, 1, 23, 23]]`
  - `[[B, n_h_lin_v, 1, 23, 64]]`
  - `[[B, n_h_lin_v, 1, 23]]`
  - `[[B, n_h_lin_v, 1, 24, 1]]`
  - `[[B, n_h_lin_v, 1, 24, 24]]`
  - `[[B, n_h_lin_v, 1, 24, 64]]`
  - `[[B, n_h_lin_v, 1, 24]]`
  - `[[B, n_h_lin_v, 1, 25, 1]]`
  - `[[B, n_h_lin_v, 1, 25, 25]]`
  - `[[B, n_h_lin_v, 1, 25, 64]]`
  - `[[B, n_h_lin_v, 1, 25]]`
  - `[[B, n_h_lin_v, 1, 26, 1]]`
  - `[[B, n_h_lin_v, 1, 26, 26]]`
  - `[[B, n_h_lin_v, 1, 26, 64]]`
  - `[[B, n_h_lin_v, 1, 26]]`
  - `[[B, n_h_lin_v, 1, 27, 1]]`
  - `[[B, n_h_lin_v, 1, 27, 27]]`
  - `[[B, n_h_lin_v, 1, 27, 64]]`
  - `[[B, n_h_lin_v, 1, 27]]`
  - `[[B, n_h_lin_v, 1, 28, 1]]`
  - `[[B, n_h_lin_v, 1, 28, 28]]`
  - `[[B, n_h_lin_v, 1, 28, 64]]`
  - `[[B, n_h_lin_v, 1, 28]]`
  - `[[B, n_h_lin_v, 1, 29, 1]]`
  - `[[B, n_h_lin_v, 1, 29, 29]]`
  - `[[B, n_h_lin_v, 1, 29, 64]]`
  - `[[B, n_h_lin_v, 1, 29]]`
  - `[[B, n_h_lin_v, 1, 2]]`
  - `[[B, n_h_lin_v, 1, 3*d_conv_lin]]`
  - `[[B, n_h_lin_v, 1, 3*n_h]]`
  - `[[B, n_h_lin_v, 1, 3*n_kv]]`
  - `[[B, n_h_lin_v, 1, 3, 1]]`
  - `[[B, n_h_lin_v, 1, 3, 3]]`
  - `[[B, n_h_lin_v, 1, 3, 64]]`
  - `[[B, n_h_lin_v, 1, 30, 1]]`
  - `[[B, n_h_lin_v, 1, 30, 30]]`
  - `[[B, n_h_lin_v, 1, 30, 64]]`
  - `[[B, n_h_lin_v, 1, 30]]`
  - `[[B, n_h_lin_v, 1, 31, 1]]`
  - `[[B, n_h_lin_v, 1, 31, 31]]`
  - `[[B, n_h_lin_v, 1, 31, 64]]`
  - `[[B, n_h_lin_v, 1, 31]]`
  - `[[B, n_h_lin_v, 1, 32, 1]]`
  - `[[B, n_h_lin_v, 1, 32, 32]]`
  - `[[B, n_h_lin_v, 1, 32, 64]]`
  - `[[B, n_h_lin_v, 1, 32]]`
  - `[[B, n_h_lin_v, 1, 33, 1]]`
  - `[[B, n_h_lin_v, 1, 33, 33]]`
  - `[[B, n_h_lin_v, 1, 33, 64]]`
  - `[[B, n_h_lin_v, 1, 33]]`
  - `[[B, n_h_lin_v, 1, 34, 1]]`
  - `[[B, n_h_lin_v, 1, 34, 34]]`
  - `[[B, n_h_lin_v, 1, 34, 64]]`
  - `[[B, n_h_lin_v, 1, 34]]`
  - `[[B, n_h_lin_v, 1, 35, 1]]`
  - `[[B, n_h_lin_v, 1, 35, 35]]`
  - `[[B, n_h_lin_v, 1, 35, 64]]`
  - `[[B, n_h_lin_v, 1, 35]]`
  - `[[B, n_h_lin_v, 1, 36, 1]]`
  - `[[B, n_h_lin_v, 1, 36, 36]]`
  - `[[B, n_h_lin_v, 1, 36, 64]]`
  - `[[B, n_h_lin_v, 1, 36]]`
  - `[[B, n_h_lin_v, 1, 37, 1]]`
  - `[[B, n_h_lin_v, 1, 37, 37]]`
  - `[[B, n_h_lin_v, 1, 37, 64]]`
  - `[[B, n_h_lin_v, 1, 37]]`
  - `[[B, n_h_lin_v, 1, 38, 1]]`
  - `[[B, n_h_lin_v, 1, 38, 38]]`
  - `[[B, n_h_lin_v, 1, 38, 64]]`
  - `[[B, n_h_lin_v, 1, 38]]`
  - `[[B, n_h_lin_v, 1, 39, 1]]`
  - `[[B, n_h_lin_v, 1, 39, 39]]`
  - `[[B, n_h_lin_v, 1, 39, 64]]`
  - `[[B, n_h_lin_v, 1, 39]]`
  - `[[B, n_h_lin_v, 1, 3]]`
  - `[[B, n_h_lin_v, 1, 4, 1]]`
  - `[[B, n_h_lin_v, 1, 4, 4]]`
  - `[[B, n_h_lin_v, 1, 4, 64]]`
  - `[[B, n_h_lin_v, 1, 40, 1]]`
  - `[[B, n_h_lin_v, 1, 40, 40]]`
  - `[[B, n_h_lin_v, 1, 40, 64]]`
  - `[[B, n_h_lin_v, 1, 40]]`
  - `[[B, n_h_lin_v, 1, 41, 1]]`
  - `[[B, n_h_lin_v, 1, 41, 41]]`
  - `[[B, n_h_lin_v, 1, 41, 64]]`
  - `[[B, n_h_lin_v, 1, 41]]`
  - `[[B, n_h_lin_v, 1, 42, 1]]`
  - `[[B, n_h_lin_v, 1, 42, 42]]`
  - `[[B, n_h_lin_v, 1, 42, 64]]`
  - `[[B, n_h_lin_v, 1, 42]]`
  - `[[B, n_h_lin_v, 1, 43, 1]]`
  - `[[B, n_h_lin_v, 1, 43, 43]]`
  - `[[B, n_h_lin_v, 1, 43, 64]]`
  - `[[B, n_h_lin_v, 1, 43]]`
  - `[[B, n_h_lin_v, 1, 44, 1]]`
  - `[[B, n_h_lin_v, 1, 44, 44]]`
  - `[[B, n_h_lin_v, 1, 44, 64]]`
  - `[[B, n_h_lin_v, 1, 44]]`
  - `[[B, n_h_lin_v, 1, 45, 1]]`
  - `[[B, n_h_lin_v, 1, 45, 45]]`
  - `[[B, n_h_lin_v, 1, 45, 64]]`
  - `[[B, n_h_lin_v, 1, 45]]`
  - `[[B, n_h_lin_v, 1, 46, 1]]`
  - `[[B, n_h_lin_v, 1, 46, 46]]`
  - `[[B, n_h_lin_v, 1, 46, 64]]`
  - `[[B, n_h_lin_v, 1, 46]]`
  - `[[B, n_h_lin_v, 1, 47, 1]]`
  - `[[B, n_h_lin_v, 1, 47, 47]]`
  - `[[B, n_h_lin_v, 1, 47, 64]]`
  - `[[B, n_h_lin_v, 1, 47]]`
  - `[[B, n_h_lin_v, 1, 48, 1]]`
  - `[[B, n_h_lin_v, 1, 48, 48]]`
  - `[[B, n_h_lin_v, 1, 48, 64]]`
  - `[[B, n_h_lin_v, 1, 48]]`
  - `[[B, n_h_lin_v, 1, 49, 1]]`
  - `[[B, n_h_lin_v, 1, 49, 49]]`
  - `[[B, n_h_lin_v, 1, 49, 64]]`
  - `[[B, n_h_lin_v, 1, 49]]`
  - `[[B, n_h_lin_v, 1, 4]]`
  - `[[B, n_h_lin_v, 1, 5, 1]]`
  - `[[B, n_h_lin_v, 1, 5, 5]]`
  - `[[B, n_h_lin_v, 1, 5, 64]]`
  - `[[B, n_h_lin_v, 1, 50, 1]]`
  - `[[B, n_h_lin_v, 1, 50, 50]]`
  - `[[B, n_h_lin_v, 1, 50, 64]]`
  - `[[B, n_h_lin_v, 1, 50]]`
  - `[[B, n_h_lin_v, 1, 51, 1]]`
  - `[[B, n_h_lin_v, 1, 51, 51]]`
  - `[[B, n_h_lin_v, 1, 51, 64]]`
  - `[[B, n_h_lin_v, 1, 51]]`
  - `[[B, n_h_lin_v, 1, 52, 1]]`
  - `[[B, n_h_lin_v, 1, 52, 52]]`
  - `[[B, n_h_lin_v, 1, 52, 64]]`
  - `[[B, n_h_lin_v, 1, 52]]`
  - `[[B, n_h_lin_v, 1, 53, 1]]`
  - `[[B, n_h_lin_v, 1, 53, 53]]`
  - `[[B, n_h_lin_v, 1, 53, 64]]`
  - `[[B, n_h_lin_v, 1, 53]]`
  - `[[B, n_h_lin_v, 1, 54, 1]]`
  - `[[B, n_h_lin_v, 1, 54, 54]]`
  - `[[B, n_h_lin_v, 1, 54, 64]]`
  - `[[B, n_h_lin_v, 1, 54]]`
  - `[[B, n_h_lin_v, 1, 55, 1]]`
  - `[[B, n_h_lin_v, 1, 55, 55]]`
  - `[[B, n_h_lin_v, 1, 55, 64]]`
  - `[[B, n_h_lin_v, 1, 55]]`
  - `[[B, n_h_lin_v, 1, 56, 1]]`
  - `[[B, n_h_lin_v, 1, 56, 56]]`
  - `[[B, n_h_lin_v, 1, 56, 64]]`
  - `[[B, n_h_lin_v, 1, 56]]`
  - `[[B, n_h_lin_v, 1, 57, 1]]`
  - `[[B, n_h_lin_v, 1, 57, 57]]`
  - `[[B, n_h_lin_v, 1, 57, 64]]`
  - `[[B, n_h_lin_v, 1, 57]]`
  - `[[B, n_h_lin_v, 1, 58, 1]]`
  - `[[B, n_h_lin_v, 1, 58, 58]]`
  - `[[B, n_h_lin_v, 1, 58, 64]]`
  - `[[B, n_h_lin_v, 1, 58]]`
  - `[[B, n_h_lin_v, 1, 59, 1]]`
  - `[[B, n_h_lin_v, 1, 59, 59]]`
  - `[[B, n_h_lin_v, 1, 59, 64]]`
  - `[[B, n_h_lin_v, 1, 59]]`
  - `[[B, n_h_lin_v, 1, 5]]`
  - `[[B, n_h_lin_v, 1, 6, 1]]`
  - `[[B, n_h_lin_v, 1, 6, 64]]`
  - `[[B, n_h_lin_v, 1, 6, 6]]`
  - `[[B, n_h_lin_v, 1, 60, 1]]`
  - `[[B, n_h_lin_v, 1, 60, 60]]`
  - `[[B, n_h_lin_v, 1, 60, 64]]`
  - `[[B, n_h_lin_v, 1, 60]]`
  - `[[B, n_h_lin_v, 1, 61, 1]]`
  - `[[B, n_h_lin_v, 1, 61, 61]]`
  - `[[B, n_h_lin_v, 1, 61, 64]]`
  - `[[B, n_h_lin_v, 1, 61]]`
  - `[[B, n_h_lin_v, 1, 62, 1]]`
  - `[[B, n_h_lin_v, 1, 62, 62]]`
  - `[[B, n_h_lin_v, 1, 62, 64]]`
  - `[[B, n_h_lin_v, 1, 62]]`
  - `[[B, n_h_lin_v, 1, 63, 1]]`
  - `[[B, n_h_lin_v, 1, 63, 63]]`
  - `[[B, n_h_lin_v, 1, 63, 64]]`
  - `[[B, n_h_lin_v, 1, 63]]`
  - `[[B, n_h_lin_v, 1, 64, 1]]`
  - `[[B, n_h_lin_v, 1, 6]]`
  - `[[B, n_h_lin_v, 1, 7, 1]]`
  - `[[B, n_h_lin_v, 1, 7, 64]]`
  - `[[B, n_h_lin_v, 1, 7, 7]]`
  - `[[B, n_h_lin_v, 1, 7]]`
  - `[[B, n_h_lin_v, 1, 8, 1]]`
  - `[[B, n_h_lin_v, 1, 8, 64]]`
  - `[[B, n_h_lin_v, 1, 8, 8]]`
  - `[[B, n_h_lin_v, 1, 8]]`
  - `[[B, n_h_lin_v, 1, 9, 1]]`
  - `[[B, n_h_lin_v, 1, 9, 64]]`
  - `[[B, n_h_lin_v, 1, 9, 9]]`
  - `[[B, n_h_lin_v, 1, 9]]`
  - `[[B, n_h_lin_v, 1, T+1]]`
  - `[[B, n_h_lin_v, 1, T]]`
  - `[[B, n_h_lin_v, 1, d_conv_lin]]`
  - `[[B, n_h_lin_v, 1, d_head_lin_k, d_rope]]`
  - `[[B, n_h_lin_v, 1, d_head_lin_k]]`
  - `[[B, n_h_lin_v, 1, d_rope, 1]]`
  - `[[B, n_h_lin_v, 1, d_rope, d_head_lin_k]]`
  - `[[B, n_h_lin_v, 1, d_rope, d_rope]]`
  - `[[B, n_h_lin_v, 1, d_rope/2]]`
  - `[[B, n_h_lin_v, 1, d_rope]]`
  - `[[B, n_h_lin_v, 1, k]]`
  - `[[B, n_h_lin_v, 1, n_h+2*n_kv]]`
  - `[[B, n_h_lin_v, 1, n_h/n_kv]]`
  - `[[B, n_h_lin_v, 1, n_h]]`
  - `[[B, n_h_lin_v, 1, n_h_lin_v+1]]`
  - `[[B, n_h_lin_v, 1, n_kv*T]]`
  - `[[B, n_h_lin_v, 1, n_kv]]`
  - `[[B, n_h_lin_v, 1]]`
  - `[[B, n_h_lin_v, T, d_head_lin_k]]`
  - `[[B, n_h_lin_v, T]]`
  - `[[B, n_h_lin_v, d_head_lin_k, 1]]`
  - `[[B, n_h_lin_v, d_head_lin_k, d_head_lin_v]]`
  - `[[B, n_h_lin_v, d_head_lin_k, d_rope]]`
  - `[[B, n_h_lin_v, d_head_lin_k]]`
  - `[[B, n_h_lin_v, d_rope, 1]]`
  - `[[B, n_h_lin_v, d_rope, d_head_lin_k]]`
  - `[[B, n_h_lin_v, d_rope, d_rope]]`
  - `[[B, n_h_lin_v, d_rope]]`
  - `[[B, n_h_lin_v]]`
  - `[[d_rope, d_rope]]`
  - `[[n_h_lin_v*T, d_head_lin_k]]`
  - `[[n_h_lin_v, d_head_lin_k, d_head_lin_v]]`
  - `[[n_h_lin_v, d_head_lin_k, d_rope]]`
  - `[[n_h_lin_v, d_head_lin_k]]`
  - `[[n_h_lin_v, d_rope, d_head_lin_k]]`
  - `[[n_h_lin_v, d_rope, d_rope]]`
  - `[[n_h_lin_v]]`
- `model.layers.*.linear_attn.conv1d`
  - `[[B, 2*n_h*d_head, n_h+2*n_kv]]`
- `model.layers.*.linear_attn.in_proj_ba`
  - `[[B, 1, d_rope]]`
  - `[[B, T, d_rope]]`
  - `[[B, d_model]]`
  - `[[B, d_rope]]`
  - `[[T, d_model]]`
  - `[[T, d_rope]]`
  - `[[d_model, d_rope]]`
- `model.layers.*.linear_attn.in_proj_qkvz`
  - `[[B, 1, 2*n_k*d_k+2*n_v*d_v]]`
  - `[[B, 2*n_k*d_k+2*n_v*d_v]]`
  - `[[B, T, 2*n_k*d_k+2*n_v*d_v]]`
  - `[[B, d_model]]`
  - `[[T, 2*n_k*d_k+2*n_v*d_v]]`
  - `[[T, d_model]]`
  - `[[d_model, 2*n_k*d_k+2*n_v*d_v]]`
- `model.layers.*.linear_attn.norm`
  - `[[n_h_lin_v*T, B]]`
  - `[[n_h_lin_v*T, d_head_lin_k]]`
  - `[[n_h_lin_v, B]]`
  - `[[n_h_lin_v, d_head_lin_k]]`
- `model.layers.*.linear_attn.out_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[B, n_h*d_head]]`
  - `[[T, d_model]]`
  - `[[T, n_h*d_head]]`
  - `[[n_h*d_head, d_model]]`
- `model.layers.*.mlp`
  - `[[B, 1, d_model]]`
  - `[[B, 1]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[T, 1]]`
  - `[[T, d_model]]`
- `model.layers.*.mlp.experts`
  - `[[B, d_model]]`
  - `[[B, k, d_model]]`
  - `[[T, d_model]]`
  - `[[T, k, d_model]]`
  - `[[d_moe, E, d_model]]`
  - `[[d_moe, d_model, 2*d_moe]]`
  - `[[d_moe]]`
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
  - `[[d_model, E]]`
- `model.layers.*.mlp.shared_expert`
  - `[[B, d_moe]]`
  - `[[T, d_moe]]`
- `model.layers.*.mlp.shared_expert.act_fn`
  - `[[B, d_moe]]`
  - `[[T, d_moe]]`
- `model.layers.*.mlp.shared_expert.down_proj`
  - `[[B, d_model]]`
  - `[[T, d_model]]`
  - `[[d_moe, d_model]]`
- `model.layers.*.mlp.shared_expert.gate_proj`
  - `[[B, d_moe]]`
  - `[[T, d_moe]]`
  - `[[d_model, d_moe]]`
- `model.layers.*.mlp.shared_expert.up_proj`
  - `[[B, d_moe]]`
  - `[[T, d_moe]]`
  - `[[d_model, d_moe]]`
- `model.layers.*.mlp.shared_expert_gate`
  - `[[B, 1]]`
  - `[[T, 1]]`
  - `[[d_model, B]]`
- `model.layers.*.post_attention_layernorm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
  - `[[d_model]]`
- `model.layers.*.self_attn`
  - `[[B, 1, 1, d_rope]]`
  - `[[B, 1, T, d_rope]]`
  - `[[B, 1, n_h*d_head]]`
  - `[[B, 1, n_h, 2*d_head]]`
  - `[[B, 1, n_h, d_head], [B, 1, n_h, d_head]]`
  - `[[B, 1, n_h, d_head]]`
  - `[[B, 1, n_kv, d_head]]`
  - `[[B, T, n_h*d_head]]`
  - `[[B, T, n_h, 2*d_head]]`
  - `[[B, T, n_h, d_head], [B, T, n_h, d_head]]`
  - `[[B, T, n_h, d_head]]`
  - `[[B, T, n_kv, d_head]]`
  - `[[B, n_h, 1, T+1]]`
  - `[[B, n_h, 1, d_head-d_rope]]`
  - `[[B, n_h, 1, d_head]]`
  - `[[B, n_h, 1, d_rope/2]]`
  - `[[B, n_h, 1, d_rope]]`
  - `[[B, n_h, T+1, d_head]]`
  - `[[B, n_h, T, T]]`
  - `[[B, n_h, T, d_head-d_rope]]`
  - `[[B, n_h, T, d_head]]`
  - `[[B, n_h, T, d_rope/2]]`
  - `[[B, n_h, T, d_rope]]`
  - `[[B, n_h, d_head, T+1]]`
  - `[[B, n_h, d_head, T]]`
  - `[[B, n_kv, 1, T+1, d_head]]`
  - `[[B, n_kv, 1, T, d_head]]`
  - `[[B, n_kv, 1, d_head-d_rope]]`
  - `[[B, n_kv, 1, d_head]]`
  - `[[B, n_kv, 1, d_rope/2]]`
  - `[[B, n_kv, 1, d_rope]]`
  - `[[B, n_kv, T+1, d_head]]`
  - `[[B, n_kv, T, d_head-d_rope]]`
  - `[[B, n_kv, T, d_head]]`
  - `[[B, n_kv, T, d_rope/2]]`
  - `[[B, n_kv, T, d_rope]]`
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
- `model.layers.*.self_attn.k_norm`
  - `[[B, 1, n_kv, 1]]`
  - `[[B, 1, n_kv, d_head]]`
  - `[[B, T, n_kv, 1]]`
  - `[[B, T, n_kv, d_head]]`
  - `[[d_head]]`
- `model.layers.*.self_attn.k_proj`
  - `[[B, 1, n_kv*d_head]]`
  - `[[B, T, n_kv*d_head]]`
  - `[[B, d_model]]`
  - `[[B, n_kv*d_head]]`
  - `[[T, d_model]]`
  - `[[T, n_kv*d_head]]`
  - `[[d_model, n_kv*d_head]]`
- `model.layers.*.self_attn.o_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[B, n_h*d_head]]`
  - `[[T, d_model]]`
  - `[[T, n_h*d_head]]`
  - `[[n_h*d_head, d_model]]`
- `model.layers.*.self_attn.q_norm`
  - `[[B, 1, n_h, 1]]`
  - `[[B, 1, n_h, d_head]]`
  - `[[B, T, n_h, 1]]`
  - `[[B, T, n_h, d_head]]`
  - `[[d_head]]`
- `model.layers.*.self_attn.q_proj`
  - `[[B, 1, 2*n_h*d_head]]`
  - `[[B, 2*n_h*d_head]]`
  - `[[B, T, 2*n_h*d_head]]`
  - `[[B, d_model]]`
  - `[[T, 2*n_h*d_head]]`
  - `[[T, d_model]]`
  - `[[d_model, 2*n_h*d_head]]`
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
  - `[[B, 1, d_rope/2]]`
  - `[[B, 1, d_rope]]`
  - `[[B, T, d_rope/2]]`
  - `[[B, T, d_rope]]`
  - `[[B, d_rope/2, 1]]`
  - `[[B, d_rope/2, T]]`
  - `[[B, d_rope/2]]`

## 이 의뢰서를 처리하는 법

`review/prompt.md` 를 LLM 에 넘기고 이 모델을 지정한다. 판정 4종과 근거 요건, 결과를 어디에 어떤 형식으로 쓰는지는 전부 `review/` 안에 있다.
