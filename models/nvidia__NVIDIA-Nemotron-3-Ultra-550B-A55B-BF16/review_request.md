# 검토 의뢰서 — nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `nemotron_h`
- 판단 필요: **1건**

## 증거 — 이미 받아둔 실제 소스

- `develop/sources/modeling_nemotron_h.py` — 있음, 이 파일을 열어서 판정한다
- `develop/sources/configuration_nemotron_h.py` — 있음, 이 파일을 열어서 판정한다

- 온라인 원본: https://github.com/huggingface/transformers/tree/main/src/transformers/models/nemotron_h

그 밖의 재료: `full/review.md`(리뷰 패킷 — shape 별 실제 행 표본), `structure.yaml`(이 모델의 심볼 표), `full/<phase>.csv`(전체 operator 표).

## 판단이 필요한 것

### 2. 이 정사각 축이 정말 같은 이름 두 번인가

`[..., X, X]` 로 렌더됐는데, 그 이름이 읽은 config 필드에서 나온 정사각 reshape 을 modeling 소스에서 찾지 못했다. 두 축 크기가 우연히 같은 것일 수 있다.

- `n_kv`

## 기계적으로 이미 확인된 것 — 다시 묻지 말 것

- **심볼이 읽은 config 필드**: 전부 이 모델의 config 클래스(또는 상속/프로퍼티/getattr 기본값)에 존재한다
- **정사각 축**: 소스에서 정사각 생성/reshape 과 대응이 확인된 축 없음
- **모듈이 읽는 config 속성**: `__init__` 에서 config 를 읽는 클래스 9개를 소스에서 확인했다. 그 목록이 각 모듈의 폭이 가질 수 있는 이름의 전부다.
- **가중치 축 ↔ 모듈 소속**: 가중치 축의 이름이 전부 그 모듈(또는 그 부모)이 실제로 읽는 config 필드에서 나왔다. 이 축들은 값이 아니라 소스로 확인된 것이다.

## 전수 점검 — 이 모델이 쓰는 이름 전부

위 절이 '풀리지 않은 것'이라면 여기는 **전부**다. 규칙이 자신 있게 붙인 이름도 틀릴 수 있고, 그런 건 미결 목록에 절대 오르지 않는다. 한 줄씩 읽고 **그 모듈에서 그 이름이 말이 되는지** 보라.

### A. 붙은 이름 전부 (28종)

| 라벨 | 값 | 나타나는 모듈 | 축 수 |
|---|---|---|---|
| `B` |  | `model.layers.*.mixer`, `model.layers.*.norm`, `model.layers.*.mixer.gate`, `model.layers.*.mixer.norm` 외 127개 | 32928 |
| `n_h_ssm` | 256 | `model.layers.*.mixer`, `model.layers.*.mixer.k_proj`, `model.layers.*.mixer.v_proj` | 14208 |
| `T` |  | `model.layers.*.mixer`, `model.layers.*.mixer.gate`, `model.layers.*.norm`, `model.layers.*.mixer.norm` 외 127개 | 12995 |
| `d_head` | 128 | `model.layers.*.mixer` | 12096 |
| `d_model` | 8192 | `model.layers.*.norm`, `model.layers.*.mixer.gate`, `model.layers.*.mixer.in_proj`, `model.layers.*.mixer.out_proj` 외 120개 | 9842 |
| `d_head_ssm` | 64 | `model.layers.*.mixer` | 5760 |
| `d_inner/n_g` |  | `model.layers.*.mixer.experts`, `model.layers.*.mixer.norm`, `model.layers.*.mixer.fc1_latent_proj`, `model.layers.*.mixer.fc2_latent_proj` | 4512 |
| `E` | 512 | `model.layers.*.mixer.gate`, `model.layers.*.mixer.experts` | 4224 |
| `k` | 22 | `model.layers.*.mixer.experts`, `model.layers.*.mixer.gate`, `model.layers.*.mixer.experts.act_fn` | 3696 |
| `d_state` | 128 | `model.layers.*.mixer` | 3168 |
| `n_g_ssm` | 8 | `model.layers.*.mixer`, `model.layers.*.mixer.norm` | 2976 |
| `d_inner` |  | `model.layers.*.mixer.norm`, `model.layers.*.mixer.out_proj`, `model.layers.*.mixer` | 2688 |
| `d_inner+2*n_g*d_state` |  | `model.layers.*.mixer`, `model.layers.*.mixer.conv1d`, `model.layers.*.mixer.act` | 2592 |
| `k*T` |  | `model.layers.*.mixer.experts`, `model.layers.*.mixer.experts.act_fn` | 2448 |
| `n_kv` | 2 | `model.layers.*.mixer`, `model.layers.*.mixer.gate` | 2208 |
| `2*d_moe` |  | `model.layers.*.mixer.shared_experts.up_proj`, `model.layers.*.mixer.shared_experts.down_proj`, `model.layers.*.mixer.shared_experts.act_fn` | 1920 |
| `d_moe` | 5120 | `model.layers.*.mixer.experts`, `model.layers.*.mixer.experts.act_fn` | 1536 |
| `n_h` | 64 | `model.layers.*.mixer` | 1512 |
| `d_conv` | 4 | `model.layers.*.mixer`, `model.layers.*.mixer.conv1d` | 1152 |
| `n_h_ssm/n_g_ssm` |  | `model.layers.*.mixer` | 960 |
| `2*d_inner+2*n_g*d_state+n_h_ssm` |  | `model.layers.*.mixer.in_proj`, `model.layers.*.mixer` | 864 |
| `n_h*d_head` |  | `model.layers.*.mixer.o_proj`, `model.layers.*.mixer.q_proj`, `model.layers.*.mixer.k_proj`, `model.layers.*.mixer.v_proj` | 792 |
| `T+1` |  | `model.layers.*.mixer`, `model` | 603 |
| `n_g*d_state` |  | `model.layers.*.mixer` | 384 |
| `d_conv+1` |  | `model.layers.*.mixer` | 144 |
| `T+d_conv-1` |  | `model.layers.*.mixer.conv1d`, `model.layers.*.mixer` | 96 |
| `d_chunk` | 128 | `model.layers.*.mixer` | 96 |
| `V` | 131072 | `lm_head`, `model.embeddings`, `(root)` | 24 |

### B. 이름 없이 남은 정수 전부 (2쌍)

**여기가 필터가 못 보던 자리다.** 정수가 남는 것 자체는 정상이다(루프 인덱스, 피연산자 개수, 브로드캐스트 축). 문제는 **이름이 있어야 하는데 없는 경우**이고, 마지막 열이 그 신호다 — 이 모델의 심볼과 값이 같다면 스코프가 그 모듈을 못 덮고 있을 수 있다. 실제로 `n_hc`(=4)가 그렇게 정수로 남아 있었다.

| 모듈 | 정수 | 축 수 | 같은 값의 심볼 |
|---|---|---|---|
| `model.layers.*.mixer` | 2 | 1944 | `n_kv` |
| `model.layers.*.mixer.gate` | 2 | 144 | `n_kv` |

### C. 모듈이 내는 출력 shape 전부 (132개 모듈 / 468종)

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
  - `[[B, 1]]`
  - `[[B, T+1]]`
  - `[[B, T]]`
  - `[[B]]`
  - `[[T+1]]`
  - `[[T]]`
  - `[[]]`
- `model.embeddings`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.*.mixer`
  - `[[2, 2]]`
  - `[[B, 1, 0], [B, 1, 0], [B, 1, d_inner], [B, 1, d_inner+2*n_g*d_state], [B, 1, n_h_ssm]]`
  - `[[B, 1, 1, d_head, n_h_ssm, d_head_ssm]]`
  - `[[B, 1, 1, d_head, n_h_ssm, d_state]]`
  - `[[B, 1, 1, n_h_ssm, d_head_ssm, d_head]]`
  - `[[B, 1, 2, d_head]]`
  - `[[B, 1, d_head, 1, n_h_ssm, d_state]]`
  - `[[B, 1, d_head, d_state, n_h_ssm, 1]]`
  - `[[B, 1, d_head, d_state, n_h_ssm, d_chunk]]`
  - `[[B, 1, d_head, d_state, n_h_ssm, d_head_ssm]]`
  - `[[B, 1, d_head, d_state, n_h_ssm]]`
  - `[[B, 1, d_head, n_h_ssm, 1, d_state]]`
  - `[[B, 1, d_head, n_h_ssm, 1]]`
  - `[[B, 1, d_head, n_h_ssm, d_head_ssm, d_state]]`
  - `[[B, 1, d_head, n_h_ssm, d_head_ssm]]`
  - `[[B, 1, d_head, n_h_ssm, d_state]]`
  - `[[B, 1, d_head, n_h_ssm]]`
  - `[[B, 1, d_inner+2*n_g*d_state]]`
  - `[[B, 1, d_inner], [B, 1, n_g*d_state], [B, 1, n_g*d_state]]`
  - `[[B, 1, d_inner]]`
  - `[[B, 1, d_model]]`
  - `[[B, 1, n_h, d_head]]`
  - `[[B, 1, n_h_ssm, d_head, 1, n_h]]`
  - `[[B, 1, n_h_ssm, d_head, d_state, 1]]`
  - `[[B, 1, n_h_ssm, d_head, d_state, n_h]]`
  - `[[B, 1, n_h_ssm, d_head, d_state]]`
  - `[[B, 1, n_h_ssm, d_head, n_h]]`
  - `[[B, 1, n_h_ssm, d_head_ssm, d_head]]`
  - `[[B, 1, n_h_ssm]]`
  - `[[B, 2, 1, T+1, d_head]]`
  - `[[B, 2, 1, T, d_head]]`
  - `[[B, 2, 1, d_head]]`
  - `[[B, 2, T+1, d_head]]`
  - `[[B, 2, T, d_head]]`
  - `[[B, 2, n_h_ssm, d_head_ssm, d_head]]`
  - `[[B, 2, n_h_ssm/n_g_ssm, T+1, d_head]]`
  - `[[B, 2, n_h_ssm/n_g_ssm, T, d_head]]`
  - `[[B, T, 0], [B, T, 0], [B, T, d_inner], [B, T, d_inner+2*n_g*d_state], [B, T, n_h_ssm]]`
  - `[[B, T, d_inner+2*n_g*d_state]]`
  - `[[B, T, d_inner], [B, T, n_g*d_state], [B, T, n_g*d_state]]`
  - `[[B, T, d_inner]]`
  - `[[B, T, d_model]]`
  - `[[B, T, n_g_ssm, 1, d_head]]`
  - `[[B, T, n_g_ssm, d_head]]`
  - `[[B, T, n_g_ssm, n_h_ssm/n_g_ssm, d_head]]`
  - `[[B, T, n_h, d_head]]`
  - `[[B, T, n_h_ssm, 1]]`
  - `[[B, T, n_h_ssm, d_head]]`
  - `[[B, T, n_h_ssm, d_head_ssm]]`
  - `[[B, T, n_h_ssm]]`
  - `[[B, T, n_kv, d_head]]`
  - `[[B, d_head, n_h_ssm, d_head_ssm]]`
  - `[[B, d_head, n_h_ssm, d_state]]`
  - `[[B, d_head, n_h_ssm]]`
  - `[[B, d_inner+2*n_g*d_state, 1]]`
  - `[[B, d_inner+2*n_g*d_state, T]]`
  - `[[B, d_inner+2*n_g*d_state, d_conv+1]]`
  - `[[B, d_inner+2*n_g*d_state, d_conv]]`
  - `[[B, d_inner+2*n_g*d_state]]`
  - `[[B, d_inner]]`
  - `[[B, d_model]]`
  - `[[B, n_g_ssm, 1, d_head]]`
  - `[[B, n_g_ssm, d_head]]`
  - `[[B, n_g_ssm, n_h_ssm/n_g_ssm, d_head]]`
  - `[[B, n_h, 1, T+1]]`
  - `[[B, n_h, 1, d_head]]`
  - `[[B, n_h, T+1, d_head]]`
  - `[[B, n_h, T, T]]`
  - `[[B, n_h, T, d_head]]`
  - `[[B, n_h, d_head, T+1]]`
  - `[[B, n_h, d_head, T]]`
  - `[[B, n_h_ssm, 1, 1]]`
  - `[[B, n_h_ssm, 1, d_head, 1]]`
  - `[[B, n_h_ssm, 1, d_head, d_state]]`
  - `[[B, n_h_ssm, 1, d_head]]`
  - `[[B, n_h_ssm, 1, n_kv, d_head_ssm, d_head]]`
  - `[[B, n_h_ssm, 1]]`
  - `[[B, n_h_ssm, d_head]]`
  - `[[B, n_h_ssm, d_head_ssm, 1]]`
  - `[[B, n_h_ssm, d_head_ssm, d_head]]`
  - `[[B, n_h_ssm, d_head_ssm]]`
  - `[[B, n_h_ssm, n_kv, 1]]`
  - `[[B, n_h_ssm, n_kv, d_head_ssm, d_head]]`
  - `[[B, n_h_ssm, n_kv, n_kv, 1, 1]]`
  - `[[B, n_h_ssm, n_kv, n_kv, 1]]`
  - `[[B, n_h_ssm, n_kv, n_kv, d_head_ssm, d_head]]`
  - `[[B, n_h_ssm, n_kv, n_kv]]`
  - `[[B, n_h_ssm, n_kv]]`
  - `[[B, n_h_ssm]]`
  - `[[T, d_model]]`
  - `[[d_head, d_state]]`
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
  - `[[n_h_ssm, d_head, B]]`
  - `[[n_h_ssm, d_head_ssm, B]]`
  - `[[n_h_ssm, d_head_ssm, d_head]]`
  - `[[n_h_ssm, d_head_ssm]]`
  - `[[n_h_ssm]]`
- `model.layers.*.mixer.act`
  - `[[B, T, d_inner+2*n_g*d_state]]`
  - `[[B, d_inner+2*n_g*d_state]]`
- `model.layers.*.mixer.conv1d`
  - `[[B, d_inner+2*n_g*d_state, T+d_conv-1]]`
- `model.layers.*.mixer.experts`
  - `[[B, d_inner/n_g]]`
  - `[[B, k, d_inner/n_g]]`
  - `[[E, d_inner/n_g, d_moe]]`
  - `[[E, d_moe, d_inner/n_g]]`
  - `[[E]]`
  - `[[T, d_inner/n_g]]`
  - `[[T, k, d_inner/n_g]]`
  - `[[k*T, B]]`
  - `[[k*T, d_inner/n_g]]`
  - `[[k*T, d_moe]]`
  - `[[k*T], [k*T]]`
  - `[[k*T]]`
  - `[[k, B]]`
  - `[[k, d_inner/n_g]]`
  - `[[k, d_moe]]`
  - `[[k], [k]]`
  - `[[k]]`
- `model.layers.*.mixer.experts.act_fn`
  - `[[k*T, d_moe]]`
  - `[[k, d_moe]]`
- `model.layers.*.mixer.fc1_latent_proj`
  - `[[B, d_inner/n_g]]`
  - `[[T, d_inner/n_g]]`
  - `[[d_model, d_inner/n_g]]`
- `model.layers.*.mixer.fc2_latent_proj`
  - `[[B, d_model]]`
  - `[[T, d_model]]`
  - `[[d_inner/n_g, d_model]]`
- `model.layers.*.mixer.gate`
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
  - `[[T, 1, E]]`
  - `[[T, 1, n_kv], [T, 1, n_kv]]`
  - `[[T, 1], [T, 1]]`
  - `[[T, 1]]`
  - `[[T, E]]`
  - `[[T, d_model]]`
  - `[[T, k], [T, k]]`
  - `[[T, k]]`
  - `[[d_model, E]]`
- `model.layers.*.mixer.in_proj`
  - `[[B, 1, 2*d_inner+2*n_g*d_state+n_h_ssm]]`
  - `[[B, 2*d_inner+2*n_g*d_state+n_h_ssm]]`
  - `[[B, T, 2*d_inner+2*n_g*d_state+n_h_ssm]]`
  - `[[B, d_model]]`
  - `[[T, 2*d_inner+2*n_g*d_state+n_h_ssm]]`
  - `[[T, d_model]]`
  - `[[d_model, 2*d_inner+2*n_g*d_state+n_h_ssm]]`
- `model.layers.*.mixer.k_proj`
  - `[[B, 1, n_h_ssm]]`
  - `[[B, T, n_h_ssm]]`
  - `[[B, d_model]]`
  - `[[B, n_h_ssm]]`
  - `[[T, d_model]]`
  - `[[T, n_h_ssm]]`
  - `[[d_model, n_h_ssm]]`
- `model.layers.*.mixer.norm`
  - `[[B, 1, d_inner]]`
  - `[[B, 1, n_g_ssm, 1]]`
  - `[[B, 1, n_g_ssm, d_inner/n_g]]`
  - `[[B, T, d_inner]]`
  - `[[B, T, n_g_ssm, 1]]`
  - `[[B, T, n_g_ssm, d_inner/n_g]]`
- `model.layers.*.mixer.o_proj`
  - `[[B, 1, n_h*d_head]]`
  - `[[B, T, n_h*d_head]]`
  - `[[B, n_h*d_head]]`
  - `[[T, n_h*d_head]]`
  - `[[n_h*d_head, n_h*d_head]]`
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
- `model.layers.*.mixer.shared_experts.act_fn`
  - `[[B, 1, 2*d_moe]]`
  - `[[B, T, 2*d_moe]]`
- `model.layers.*.mixer.shared_experts.down_proj`
  - `[[2*d_moe, d_model]]`
  - `[[B, 1, d_model]]`
  - `[[B, 2*d_moe]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[T, 2*d_moe]]`
  - `[[T, d_model]]`
- `model.layers.*.mixer.shared_experts.up_proj`
  - `[[B, 1, 2*d_moe]]`
  - `[[B, 2*d_moe]]`
  - `[[B, T, 2*d_moe]]`
  - `[[B, d_model]]`
  - `[[T, 2*d_moe]]`
  - `[[T, d_model]]`
  - `[[d_model, 2*d_moe]]`
- `model.layers.*.mixer.v_proj`
  - `[[B, 1, n_h_ssm]]`
  - `[[B, T, n_h_ssm]]`
  - `[[B, d_model]]`
  - `[[B, n_h_ssm]]`
  - `[[T, d_model]]`
  - `[[T, n_h_ssm]]`
  - `[[d_model, n_h_ssm]]`
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
- `model.norm_f`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`

## 이 의뢰서를 처리하는 법

`review/prompt.md` 를 LLM 에 넘기고 이 모델을 지정한다. 판정 4종과 근거 요건, 결과를 어디에 어떤 형식으로 쓰는지는 전부 `review/` 안에 있다.
