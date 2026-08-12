# 검토 의뢰서 — nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `nemotron_h`
- 판단 필요: **1건**

## 증거 — 이미 받아둔 실제 소스

- `develop/sources/modeling_nemotron_h.py` — 있음, 이 파일을 열어서 판정한다
- `develop/sources/configuration_nemotron_h.py` — 있음, 이 파일을 열어서 판정한다

- 온라인 원본: https://github.com/huggingface/transformers/tree/main/src/transformers/models/nemotron_h

그 밖의 재료: `full/review.md`(리뷰 패킷 — shape 별 실제 행 표본), `structure.yaml`(이 모델의 심볼 표), `full/<phase>.csv`(전체 operator 표).

## 판단이 필요한 것

### 6. 값이 겹쳐 **임의로** 고른 축

두 심볼이 같은 값을 갖는 자리다. 규칙에는 고를 근거가 없고, 이긴 쪽은 전역 우선순위 — 즉 **관례**로 정해졌다. 이름이 맞을 수도 있지만 파이프라인은 그걸 알지 못한다. 표에서는 확신 있는 라벨과 똑같이 보인다.

**소스를 열어 어느 쪽인지 확정하는 것이 여기서 할 일이다.** 확정되면 `rules/label_overrides.yaml` 에 근거와 함께 못 박는다(review/05-overrides.md). 출신으로만 구별되는 경우라면 그렇게 적고 `open` 으로 남긴다.

- `d_head vs d_state` in `model.layers.*.mixer` — 값 128 를 두고 후보가 2개, 528축

## 기계적으로 이미 확인된 것 — 다시 묻지 말 것

- **심볼이 읽은 config 필드**: 전부 이 모델의 config 클래스(또는 상속/프로퍼티/getattr 기본값)에 존재한다
- **정사각 축**: `d_chunk` ← 소스의 `chunk_size` ← `chunk_size`
- **모듈이 읽는 config 속성**: `__init__` 에서 config 를 읽는 클래스 9개를 소스에서 확인했다. 그 목록이 각 모듈의 폭이 가질 수 있는 이름의 전부다.
- **가중치 축 ↔ 모듈 소속**: 가중치 축의 이름이 전부 그 모듈(또는 그 부모)이 실제로 읽는 config 필드에서 나왔다. 이 축들은 값이 아니라 소스로 확인된 것이다.

## 전수 점검 — 이 모델이 쓰는 이름 전부

위 절이 '풀리지 않은 것'이라면 여기는 **전부**다. 규칙이 자신 있게 붙인 이름도 틀릴 수 있고, 그런 건 미결 목록에 절대 오르지 않는다. 한 줄씩 읽고 **그 모듈에서 그 이름이 말이 되는지** 보라.

### A. 붙은 이름 전부 (23종)

| 라벨 | 값 | 나타나는 모듈 | 축 수 |
|---|---|---|---|
| `B` |  | `model.layers.*.mixer`, `model.layers.*.norm`, `model.layers.*.mixer.norm`, `model.layers.*.mixer.in_proj` 외 56개 | 12089 |
| `n_h_ssm` | 96 | `model.layers.*.mixer` | 6027 |
| `T` |  | `model.layers.*.mixer`, `model.layers.*.norm`, `model.layers.*.mixer.norm`, `model.layers.*.mixer.in_proj` 외 56개 | 3911 |
| `d_chunk` | 256 | `model.layers.*.mixer` | 3381 |
| `d_model` | 3136 | `model.layers.*.norm`, `model.layers.*.mixer.in_proj`, `model.layers.*.mixer.out_proj`, `model.layers.*.mixer.up_proj` 외 51개 | 2782 |
| `d_state` | 128 | `model.layers.*.mixer` | 2772 |
| `d_head_ssm` | 80 | `model.layers.*.mixer` | 2688 |
| `n_g_ssm` | 8 | `model.layers.*.mixer`, `model.layers.*.mixer.norm` | 1534 |
| `d_inner` |  | `model.layers.*.mixer.norm`, `model.layers.*.mixer.out_proj`, `model.layers.*.mixer` | 1176 |
| `d_inner+2*n_g*d_state` |  | `model.layers.*.mixer`, `model.layers.*.mixer.conv1d`, `model.layers.*.mixer.act` | 1134 |
| `d_ff` | 12544 | `model.layers.*.mixer.up_proj`, `model.layers.*.mixer.down_proj`, `model.layers.*.mixer.act_fn` | 680 |
| `d_head` | 128 | `model.layers.*.mixer` | 528 |
| `d_conv` | 4 | `model.layers.*.mixer`, `model.layers.*.mixer.conv1d` | 504 |
| `n_h` | 40 | `model.layers.*.mixer` | 400 |
| `2*d_inner+2*n_g*d_state+n_h_ssm` |  | `model.layers.*.mixer.in_proj`, `model.layers.*.mixer` | 378 |
| `n_h_ssm/n_g_ssm` |  | `model.layers.*.mixer` | 336 |
| `n_g*d_state` |  | `model.layers.*.mixer`, `model.layers.*.mixer.k_proj`, `model.layers.*.mixer.v_proj` | 312 |
| `d_inner/n_g` |  | `model.layers.*.mixer.norm` | 294 |
| `T+1` |  | `model.layers.*.mixer` | 192 |
| `n_h*d_head` |  | `model.layers.*.mixer.q_proj`, `model.layers.*.mixer.o_proj`, `model.layers.*.mixer` | 144 |
| `n_h/n_g_ssm` |  | `model.layers.*.mixer` | 127 |
| `T+d_conv-1` |  | `model.layers.*.mixer.conv1d`, `model.layers.*.mixer` | 42 |
| `V` | 131072 | `lm_head`, `model.embeddings`, `(root)` | 24 |

### B. 이름 없이 남은 정수 전부 (1쌍)

**여기가 필터가 못 보던 자리다.** 정수가 남는 것 자체는 정상이다(루프 인덱스, 피연산자 개수, 브로드캐스트 축). 문제는 **이름이 있어야 하는데 없는 경우**이고, 마지막 열이 그 신호다 — 이 모델의 심볼과 값이 같다면 스코프가 그 모듈을 못 덮고 있을 수 있다. 실제로 `n_hc`(=4)가 그렇게 정수로 남아 있었다.

| 모듈 | 정수 | 축 수 | 같은 값의 심볼 |
|---|---|---|---|
| `model.layers.*.mixer` | 2 | 1491 | `k` |

### C. 모듈이 내는 출력 shape 전부 (60개 모듈 / 280종)

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
- `model.embeddings`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.*.mixer`
  - `[[2, 2]]`
  - `[[B, 1, 0], [B, 1, 0], [B, 1, d_inner], [B, 1, d_inner+2*n_g*d_state], [B, 1, n_h_ssm]]`
  - `[[B, 1, 1, d_chunk, n_h_ssm, d_head_ssm]]`
  - `[[B, 1, 1, d_chunk, n_h_ssm, d_state]]`
  - `[[B, 1, 1, n_h_ssm, d_head_ssm, d_state]]`
  - `[[B, 1, d_chunk, 1, n_h_ssm, d_state]]`
  - `[[B, 1, d_chunk, d_chunk, n_h_ssm, 1]]`
  - `[[B, 1, d_chunk, d_chunk, n_h_ssm, d_head_ssm]]`
  - `[[B, 1, d_chunk, d_chunk, n_h_ssm, d_state]]`
  - `[[B, 1, d_chunk, d_chunk, n_h_ssm]]`
  - `[[B, 1, d_chunk, n_h_ssm, 1, d_state]]`
  - `[[B, 1, d_chunk, n_h_ssm, 1]]`
  - `[[B, 1, d_chunk, n_h_ssm, d_head_ssm, d_state]]`
  - `[[B, 1, d_chunk, n_h_ssm, d_head_ssm]]`
  - `[[B, 1, d_chunk, n_h_ssm, d_state]]`
  - `[[B, 1, d_chunk, n_h_ssm]]`
  - `[[B, 1, d_inner+2*n_g*d_state]]`
  - `[[B, 1, d_inner], [B, 1, n_g*d_state], [B, 1, n_g*d_state]]`
  - `[[B, 1, d_inner]]`
  - `[[B, 1, n_g_ssm, d_head]]`
  - `[[B, 1, n_h*d_head]]`
  - `[[B, 1, n_h, d_head]]`
  - `[[B, 1, n_h_ssm, d_chunk, 1, d_head_ssm]]`
  - `[[B, 1, n_h_ssm, d_chunk, d_head_ssm]]`
  - `[[B, 1, n_h_ssm, d_chunk, d_state, 1]]`
  - `[[B, 1, n_h_ssm, d_chunk, d_state, d_head_ssm]]`
  - `[[B, 1, n_h_ssm, d_chunk, d_state]]`
  - `[[B, 1, n_h_ssm, d_head_ssm, d_state]]`
  - `[[B, 1, n_h_ssm, d_state, d_head_ssm]]`
  - `[[B, 1, n_h_ssm]]`
  - `[[B, 2, n_h_ssm, d_head_ssm, d_state]]`
  - `[[B, T, 0], [B, T, 0], [B, T, d_inner], [B, T, d_inner+2*n_g*d_state], [B, T, n_h_ssm]]`
  - `[[B, T, d_inner+2*n_g*d_state]]`
  - `[[B, T, d_inner], [B, T, n_g*d_state], [B, T, n_g*d_state]]`
  - `[[B, T, d_inner]]`
  - `[[B, T, n_g_ssm, 1, d_state]]`
  - `[[B, T, n_g_ssm, d_head]]`
  - `[[B, T, n_g_ssm, d_state]]`
  - `[[B, T, n_g_ssm, n_h_ssm/n_g_ssm, d_state]]`
  - `[[B, T, n_h*d_head]]`
  - `[[B, T, n_h, d_head]]`
  - `[[B, T, n_h_ssm, 1]]`
  - `[[B, T, n_h_ssm, d_head_ssm]]`
  - `[[B, T, n_h_ssm, d_state]]`
  - `[[B, T, n_h_ssm]]`
  - `[[B, d_chunk, n_h_ssm, d_head_ssm]]`
  - `[[B, d_chunk, n_h_ssm, d_state]]`
  - `[[B, d_chunk, n_h_ssm]]`
  - `[[B, d_inner+2*n_g*d_state, 1]]`
  - `[[B, d_inner+2*n_g*d_state, T]]`
  - `[[B, d_inner+2*n_g*d_state, d_conv]]`
  - `[[B, d_inner+2*n_g*d_state, n_h/n_g_ssm]]`
  - `[[B, d_inner+2*n_g*d_state]]`
  - `[[B, d_inner]]`
  - `[[B, n_g_ssm, 1, T+1, d_head]]`
  - `[[B, n_g_ssm, 1, T, d_head]]`
  - `[[B, n_g_ssm, 1, d_head]]`
  - `[[B, n_g_ssm, 1, d_state]]`
  - `[[B, n_g_ssm, T+1, d_head]]`
  - `[[B, n_g_ssm, T, d_head]]`
  - `[[B, n_g_ssm, d_state]]`
  - `[[B, n_g_ssm, n_h/n_g_ssm, T+1, d_head]]`
  - `[[B, n_g_ssm, n_h/n_g_ssm, T, d_head]]`
  - `[[B, n_g_ssm, n_h_ssm/n_g_ssm, d_state]]`
  - `[[B, n_h, 1, T+1]]`
  - `[[B, n_h, 1, d_head]]`
  - `[[B, n_h, T+1, d_head]]`
  - `[[B, n_h, T, T]]`
  - `[[B, n_h, T, d_head]]`
  - `[[B, n_h, d_head, T+1]]`
  - `[[B, n_h, d_head, T]]`
  - `[[B, n_h_ssm, 1, 1]]`
  - `[[B, n_h_ssm, 1, 2, d_head_ssm, d_state]]`
  - `[[B, n_h_ssm, 1, d_chunk, 1]]`
  - `[[B, n_h_ssm, 1, d_chunk, d_chunk]]`
  - `[[B, n_h_ssm, 1, d_chunk]]`
  - `[[B, n_h_ssm, 1, d_state]]`
  - `[[B, n_h_ssm, 1]]`
  - `[[B, n_h_ssm, 2, 1]]`
  - `[[B, n_h_ssm, 2, 2, 1, 1]]`
  - `[[B, n_h_ssm, 2, 2, 1]]`
  - `[[B, n_h_ssm, 2, 2, d_head_ssm, d_state]]`
  - `[[B, n_h_ssm, 2, 2]]`
  - `[[B, n_h_ssm, 2, d_head_ssm, d_state]]`
  - `[[B, n_h_ssm, 2]]`
  - `[[B, n_h_ssm, d_head_ssm, 1]]`
  - `[[B, n_h_ssm, d_head_ssm, d_state]]`
  - `[[B, n_h_ssm, d_head_ssm]]`
  - `[[B, n_h_ssm, d_state]]`
  - `[[B, n_h_ssm]]`
  - `[[T, T]]`
  - `[[]]`
  - `[[d_chunk, d_chunk]]`
  - `[[d_inner+2*n_g*d_state, d_conv]]`
  - `[[n_h, B, T+1]]`
  - `[[n_h, B, d_head]]`
  - `[[n_h, T+1, d_head]]`
  - `[[n_h, T, T]]`
  - `[[n_h, T, d_head]]`
  - `[[n_h, d_head, T+1]]`
  - `[[n_h, d_head, T]]`
  - `[[n_h_ssm, B, 1]]`
  - `[[n_h_ssm, B]]`
  - `[[n_h_ssm, d_head_ssm, B]]`
  - `[[n_h_ssm, d_head_ssm, d_state]]`
  - `[[n_h_ssm, d_head_ssm]]`
  - `[[n_h_ssm, d_state, B]]`
  - `[[n_h_ssm]]`
- `model.layers.*.mixer.act`
  - `[[B, T, d_inner+2*n_g*d_state]]`
  - `[[B, d_inner+2*n_g*d_state]]`
- `model.layers.*.mixer.act_fn`
  - `[[B, 1, d_ff]]`
  - `[[B, T, d_ff]]`
- `model.layers.*.mixer.conv1d`
  - `[[B, d_inner+2*n_g*d_state, T+d_conv-1]]`
- `model.layers.*.mixer.down_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_ff]]`
  - `[[B, d_model]]`
  - `[[T, d_ff]]`
  - `[[T, d_model]]`
  - `[[d_ff, d_model]]`
- `model.layers.*.mixer.in_proj`
  - `[[B, 1, 2*d_inner+2*n_g*d_state+n_h_ssm]]`
  - `[[B, 2*d_inner+2*n_g*d_state+n_h_ssm]]`
  - `[[B, T, 2*d_inner+2*n_g*d_state+n_h_ssm]]`
  - `[[B, d_model]]`
  - `[[T, 2*d_inner+2*n_g*d_state+n_h_ssm]]`
  - `[[T, d_model]]`
  - `[[d_model, 2*d_inner+2*n_g*d_state+n_h_ssm]]`
- `model.layers.*.mixer.k_proj`
  - `[[B, 1, n_g*d_state]]`
  - `[[B, T, n_g*d_state]]`
  - `[[B, d_model]]`
  - `[[B, n_g*d_state]]`
  - `[[T, d_model]]`
  - `[[T, n_g*d_state]]`
  - `[[d_model, n_g*d_state]]`
- `model.layers.*.mixer.norm`
  - `[[B, 1, d_inner]]`
  - `[[B, 1, n_g_ssm, 1]]`
  - `[[B, 1, n_g_ssm, d_inner/n_g]]`
  - `[[B, T, d_inner]]`
  - `[[B, T, n_g_ssm, 1]]`
  - `[[B, T, n_g_ssm, d_inner/n_g]]`
- `model.layers.*.mixer.o_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[B, n_h*d_head]]`
  - `[[T, d_model]]`
  - `[[T, n_h*d_head]]`
  - `[[n_h*d_head, d_model]]`
- `model.layers.*.mixer.out_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_inner]]`
  - `[[B, d_model]]`
  - `[[T, d_inner]]`
  - `[[T, d_model]]`
  - `[[d_inner, d_model]]`
- `model.layers.*.mixer.q_proj`
  - `[[B, 1, n_h*d_head]]`
  - `[[B, T, n_h*d_head]]`
  - `[[B, d_model]]`
  - `[[B, n_h*d_head]]`
  - `[[T, d_model]]`
  - `[[T, n_h*d_head]]`
  - `[[d_model, n_h*d_head]]`
- `model.layers.*.mixer.up_proj`
  - `[[B, 1, d_ff]]`
  - `[[B, T, d_ff]]`
  - `[[B, d_ff]]`
  - `[[B, d_model]]`
  - `[[T, d_ff]]`
  - `[[T, d_model]]`
  - `[[d_model, d_ff]]`
- `model.layers.*.mixer.v_proj`
  - `[[B, 1, n_g*d_state]]`
  - `[[B, T, n_g*d_state]]`
  - `[[B, d_model]]`
  - `[[B, n_g*d_state]]`
  - `[[T, d_model]]`
  - `[[T, n_g*d_state]]`
  - `[[d_model, n_g*d_state]]`
- `model.layers.*.norm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
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
- `model.norm_f`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`

## 이 의뢰서를 처리하는 법

`review/prompt.md` 를 LLM 에 넘기고 이 모델을 지정한다. 판정 4종과 근거 요건, 결과를 어디에 어떤 형식으로 쓰는지는 전부 `review/` 안에 있다.
