# 검토 의뢰서 — allenai/OLMoE-1B-7B-0924

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `olmoe`
- 판단 필요: **1건**

## 증거 — 이미 받아둔 실제 소스

- `develop/sources/modeling_olmoe.py` — 있음, 이 파일을 열어서 판정한다
- `develop/sources/configuration_olmoe.py` — 있음, 이 파일을 열어서 판정한다

- 온라인 원본: https://github.com/huggingface/transformers/tree/main/src/transformers/models/olmoe

그 밖의 재료: `full/review.md`(리뷰 패킷 — shape 별 실제 행 표본), `structure.yaml`(이 모델의 심볼 표), `full/<phase>.csv`(전체 operator 표).

## 판단이 필요한 것

### 6. 값이 겹쳐 **임의로** 고른 축

두 심볼이 같은 값을 갖는 자리다. 규칙에는 고를 근거가 없고, 이긴 쪽은 전역 우선순위 — 즉 **관례**로 정해졌다. 이름이 맞을 수도 있지만 파이프라인은 그걸 알지 못한다. 표에서는 확신 있는 라벨과 똑같이 보인다.

**소스를 열어 어느 쪽인지 확정하는 것이 여기서 할 일이다.** 확정되면 `rules/label_overrides.yaml` 에 근거와 함께 못 박는다(review/05-overrides.md). 출신으로만 구별되는 경우라면 그렇게 적고 `open` 으로 남긴다.

- `n_h vs n_kv` in `model.layers.*.self_attn` — 값 16 를 두고 후보가 2개, 3040축

## 기계적으로 이미 확인된 것 — 다시 묻지 말 것

- **심볼이 읽은 config 필드**: 전부 이 모델의 config 클래스(또는 상속/프로퍼티/getattr 기본값)에 존재한다
- **정사각 축**: 소스에서 정사각 생성/reshape 과 대응이 확인된 축 없음
- **모듈이 읽는 config 속성**: `__init__` 에서 config 를 읽는 클래스 8개를 소스에서 확인했다. 그 목록이 각 모듈의 폭이 가질 수 있는 이름의 전부다.
- **가중치 축 ↔ 모듈 소속**: 가중치 축의 이름이 전부 그 모듈(또는 그 부모)이 실제로 읽는 config 필드에서 나왔다. 이 축들은 값이 아니라 소스로 확인된 것이다.

## 전수 점검 — 이 모델이 쓰는 이름 전부

위 절이 '풀리지 않은 것'이라면 여기는 **전부**다. 규칙이 자신 있게 붙인 이름도 틀릴 수 있고, 그런 건 미결 목록에 절대 오르지 않는다. 한 줄씩 읽고 **그 모듈에서 그 이름이 말이 되는지** 보라.

### A. 붙은 이름 전부 (14종)

| 라벨 | 값 | 나타나는 모듈 | 축 수 |
|---|---|---|---|
| `B` |  | `model.layers.*.self_attn`, `model.layers.*.input_layernorm`, `model.layers.*.self_attn.q_norm`, `model.layers.*.self_attn.k_norm` 외 29개 | 6736 |
| `T` |  | `model.layers.*.self_attn`, `model.layers.*.input_layernorm`, `model.layers.*.self_attn.q_norm`, `model.layers.*.self_attn.k_norm` 외 29개 | 4168 |
| `d_model` | 2048 | `model.layers.*.mlp.experts`, `model.layers.*.input_layernorm`, `model.layers.*.post_attention_layernorm`, `model.layers.*.self_attn.q_proj` 외 24개 | 3154 |
| `n_h` | 16 | `model.layers.*.self_attn` | 3040 |
| `d_head` | 128 | `model.layers.*.self_attn`, `model.rotary_emb` | 2522 |
| `n_h*d_head` |  | `model.layers.*.self_attn.o_proj`, `model.layers.*.self_attn.q_norm`, `model.layers.*.self_attn.k_norm`, `model.layers.*.self_attn.q_proj` 외 3개 | 2176 |
| `k` | 8 | `model.layers.*.mlp.experts`, `model.layers.*.mlp.gate`, `model.layers.*.mlp.experts.act_fn` | 1136 |
| `k*T` |  | `model.layers.*.mlp.experts`, `model.layers.*.mlp.experts.act_fn` | 880 |
| `E` | 64 | `model.layers.*.mlp.experts`, `model.layers.*.mlp.gate` | 832 |
| `T+1` |  | `model.layers.*.self_attn` | 512 |
| `d_head/2` |  | `model.layers.*.self_attn`, `model.rotary_emb` | 420 |
| `d_moe` | 1024 | `model.layers.*.mlp.experts`, `model.layers.*.mlp.experts.act_fn` | 416 |
| `2*d_moe` |  | `model.layers.*.mlp.experts` | 64 |
| `V` | 50304 | `lm_head`, `model.embed_tokens` | 20 |

### B. 이름 없이 남은 정수 전부 (0쌍)

**여기가 필터가 못 보던 자리다.** 정수가 남는 것 자체는 정상이다(루프 인덱스, 피연산자 개수, 브로드캐스트 축). 문제는 **이름이 있어야 하는데 없는 경우**이고, 마지막 열이 그 신호다 — 이 모델의 심볼과 값이 같다면 스코프가 그 모듈을 못 덮고 있을 수 있다. 실제로 `n_hc`(=4)가 그렇게 정수로 남아 있었다.

| 모듈 | 정수 | 축 수 | 같은 값의 심볼 |
|---|---|---|---|

### C. 모듈이 내는 출력 shape 전부 (34개 모듈 / 158종)

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
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[T, d_model]]`
- `model.layers.*.mlp.experts`
  - `[[B, d_model]]`
  - `[[B, k, d_model]]`
  - `[[E, d_model, d_model]]`
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
  - `[[B, E]]`
  - `[[B, d_model]]`
  - `[[B, k], [B, k]]`
  - `[[B, k]]`
  - `[[T, E]]`
  - `[[T, d_model]]`
  - `[[T, k], [T, k]]`
  - `[[T, k]]`
  - `[[d_model, E]]`
- `model.layers.*.post_attention_layernorm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
- `model.layers.*.self_attn`
  - `[[B, 1, 1, d_head]]`
  - `[[B, 1, T, d_head]]`
  - `[[B, 1, n_h*d_head]]`
  - `[[B, 1, n_h, d_head]]`
  - `[[B, T, n_h*d_head]]`
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
- `model.layers.*.self_attn.k_norm`
  - `[[B, 1, 1]]`
  - `[[B, 1, n_h*d_head]]`
  - `[[B, T, 1]]`
  - `[[B, T, n_h*d_head]]`
- `model.layers.*.self_attn.k_proj`
  - `[[B, 1, n_h*d_head]]`
  - `[[B, T, n_h*d_head]]`
  - `[[B, d_model]]`
  - `[[B, n_h*d_head]]`
  - `[[T, d_model]]`
  - `[[T, n_h*d_head]]`
  - `[[d_model, n_h*d_head]]`
- `model.layers.*.self_attn.o_proj`
  - `[[B, 1, n_h*d_head]]`
  - `[[B, T, n_h*d_head]]`
  - `[[B, n_h*d_head]]`
  - `[[T, n_h*d_head]]`
  - `[[n_h*d_head, n_h*d_head]]`
- `model.layers.*.self_attn.q_norm`
  - `[[B, 1, 1]]`
  - `[[B, 1, n_h*d_head]]`
  - `[[B, T, 1]]`
  - `[[B, T, n_h*d_head]]`
- `model.layers.*.self_attn.q_proj`
  - `[[B, 1, n_h*d_head]]`
  - `[[B, T, n_h*d_head]]`
  - `[[B, d_model]]`
  - `[[B, n_h*d_head]]`
  - `[[T, d_model]]`
  - `[[T, n_h*d_head]]`
  - `[[d_model, n_h*d_head]]`
- `model.layers.*.self_attn.v_proj`
  - `[[B, 1, n_h*d_head]]`
  - `[[B, T, n_h*d_head]]`
  - `[[B, d_model]]`
  - `[[B, n_h*d_head]]`
  - `[[T, d_model]]`
  - `[[T, n_h*d_head]]`
  - `[[d_model, n_h*d_head]]`
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
