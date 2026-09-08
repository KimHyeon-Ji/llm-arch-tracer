# 검토 의뢰서 — moonshotai/Kimi-Linear-48B-A3B-Instruct

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `kimi_linear`
- 판단 필요: **65건**

## 증거 — 이미 받아둔 실제 소스

- `develop/sources/modeling_kimi_linear.py` — 있음, 이 파일을 열어서 판정한다
- `develop/sources/configuration_kimi_linear.py` — 있음, 이 파일을 열어서 판정한다

- 온라인 원본: https://github.com/huggingface/transformers/tree/main/src/transformers/models/kimi_linear

그 밖의 재료: `full/review.md`(리뷰 패킷 — shape 별 실제 행 표본), `structure.yaml`(이 모델의 심볼 표), `full/<phase>.csv`(전체 operator 표).

## 판단이 필요한 것

### 1. 이 config 필드가 정말 이 뜻인가

값은 로드된 config 에 있지만 이 모델의 config 클래스가 선언한 필드가 아니다 (체크포인트 `config.json` 에서 온 값). 클래스가 뜻을 보증하지 않으므로 modeling 소스에서 이 필드가 실제로 어떻게 쓰이는지 확인해야 한다.

- `n_h_kda ← linear_attn_config`
- `d_head_kda ← linear_attn_config`
- `d_conv ← linear_attn_config`

### 2. 이 정사각 축이 정말 같은 이름 두 번인가

`[..., X, X]` 로 렌더됐는데, 그 이름이 읽은 config 필드에서 나온 정사각 reshape 을 modeling 소스에서 찾지 못했다. 두 축 크기가 우연히 같은 것일 수 있다.

- `d_head_kda`

### 4. 규칙 없이 산술로 지은 이름

값이 맞아떨어져서 붙인 이름이다. 산술적으로 참이어도 틀린 이름일 수 있으므로 (예: RoPE 절반 차원) 소스에서 확인이 필요하다.

- `4*k` in `model.layers.*.block_sparse_moe.experts.0 (레이어 3개)` — heur_multiple, 18축
- `4*k` in `model.layers.*.block_sparse_moe.experts.1 (레이어 3개)` — heur_multiple, 18축
- `4*k` in `model.layers.*.block_sparse_moe.experts.2 (레이어 3개)` — heur_multiple, 18축
- `4*k` in `model.layers.*.block_sparse_moe.experts.3 (레이어 3개)` — heur_multiple, 18축

### 6. 값이 겹쳐 **임의로** 고른 축

두 심볼이 같은 값을 갖는 자리다. 규칙에는 고를 근거가 없고, 이긴 쪽은 전역 우선순위 — 즉 **관례**로 정해졌다. 이름이 맞을 수도 있지만 파이프라인은 그걸 알지 못한다. 표에서는 확신 있는 라벨과 똑같이 보인다.

**소스를 열어 어느 쪽인지 확정하는 것이 여기서 할 일이다.** 확정되면 `rules/label_overrides.yaml` 에 근거와 함께 못 박는다(review/05-overrides.md). 출신으로만 구별되는 경우라면 그렇게 적고 `open` 으로 남긴다.

- `n_h vs n_kv` in `model.layers.*.self_attn` — 값 32 를 두고 후보가 2개, 896축
- `d_nope vs d_v` in `model.layers.*.self_attn` — 값 128 를 두고 후보가 2개, 259축

### 5. 그 모듈이 읽지도 않는 필드의 이름이 가중치 축에 붙어 있다

**값을 전혀 보지 않는 안건이다.** 파라미터의 shape 은 그것을 소유한 모듈이 선언한 것이므로, 그 모듈도 그 모듈을 만든 부모도 읽지 않는 config 필드의 이름이 붙어 있으면 산술이 맞아도 근거가 없다. 소스에서 그 `nn.Linear`/`nn.Parameter` 를 만드는 줄을 찾아 실제 폭이 무엇인지 확인하라.

- `d_moe` (← config `moe_intermediate_size/moe_shared_expert_intermediate_size/shared_expert_intermediate_size`) in `model.layers.*.block_sparse_moe.experts.*.w1` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 416축
- `d_moe` (← config `moe_intermediate_size/moe_shared_expert_intermediate_size/shared_expert_intermediate_size`) in `model.layers.*.block_sparse_moe.experts.*.w2` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 416축
- `d_moe` (← config `moe_intermediate_size/moe_shared_expert_intermediate_size/shared_expert_intermediate_size`) in `model.layers.*.block_sparse_moe.experts.*.w3` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 416축
- `d_head_kda` (← config `linear_attn_config`) in `model.layers.*.self_attn.f_b_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 160축
- `d_head_kda` (← config `linear_attn_config`) in `model.layers.*.self_attn.g_b_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 160축
- `E` (← config `n_routed_experts/num_experts/num_local_experts`) in `model.layers.*.block_sparse_moe.gate` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 156축
- `d_moe` (← config `moe_intermediate_size/moe_shared_expert_intermediate_size/shared_expert_intermediate_size`) in `model.layers.*.block_sparse_moe.shared_experts.down_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 104축
- `d_moe` (← config `moe_intermediate_size/moe_shared_expert_intermediate_size/shared_expert_intermediate_size`) in `model.layers.*.block_sparse_moe.shared_experts.gate_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 104축
- `d_moe` (← config `moe_intermediate_size/moe_shared_expert_intermediate_size/shared_expert_intermediate_size`) in `model.layers.*.block_sparse_moe.shared_experts.up_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 104축
- `n_h_kda` (← config `linear_attn_config`) in `model.layers.*.self_attn` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 80축
- `n_h_kda` (← config `linear_attn_config`) in `model.layers.*.self_attn.b_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 80축
- `d_head_kda` (← config `linear_attn_config`) in `model.layers.*.self_attn.f_a_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 80축
- `n_h_kda` (← config `linear_attn_config`) in `model.layers.*.self_attn.f_b_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 80축
- `d_head_kda` (← config `linear_attn_config`) in `model.layers.*.self_attn.g_a_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 80축
- `n_h_kda` (← config `linear_attn_config`) in `model.layers.*.self_attn.g_b_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 80축
- `d_head_kda` (← config `linear_attn_config`) in `model.layers.*.self_attn.k_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 80축
- `n_h_kda` (← config `linear_attn_config`) in `model.layers.*.self_attn.k_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 80축
- `d_head_kda` (← config `linear_attn_config`) in `model.layers.*.self_attn.o_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 80축
- `n_h_kda` (← config `linear_attn_config`) in `model.layers.*.self_attn.o_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 80축
- `d_head_kda` (← config `linear_attn_config`) in `model.layers.*.self_attn.q_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 80축
- `n_h_kda` (← config `linear_attn_config`) in `model.layers.*.self_attn.q_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 80축
- `d_head_kda` (← config `linear_attn_config`) in `model.layers.*.self_attn.v_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 80축
- `n_h_kda` (← config `linear_attn_config`) in `model.layers.*.self_attn.v_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 80축
- `c_kv` (← config `kv_lora_rank`) in `model.layers.*.self_attn.kv_a_proj_with_mqa` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 28축
- `d_rope` (← config `qk_rope_head_dim`) in `model.layers.*.self_attn.kv_a_proj_with_mqa` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 28축
- `c_kv` (← config `kv_lora_rank`) in `model.layers.*.self_attn.kv_b_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 28축
- `d_nope` (← config `qk_nope_head_dim`) in `model.layers.*.self_attn.kv_b_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 28축
- `d_v` (← config `v_head_dim`) in `model.layers.*.self_attn.kv_b_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 28축
- `n_h` (← config `n_head/num_attention_heads/num_heads`) in `model.layers.*.self_attn.kv_b_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 28축
- `d_v` (← config `v_head_dim`) in `model.layers.*.self_attn.o_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 28축
- `n_h` (← config `n_head/num_attention_heads/num_heads`) in `model.layers.*.self_attn.o_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 28축
- `d_nope` (← config `qk_nope_head_dim`) in `model.layers.*.self_attn.q_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 28축
- `d_rope` (← config `qk_rope_head_dim`) in `model.layers.*.self_attn.q_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 28축
- `n_h` (← config `n_head/num_attention_heads/num_heads`) in `model.layers.*.self_attn.q_proj` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 28축
- `d_conv` (← config `conv_L_cache/conv_kernel/d_conv/linear_attn_config/mamba_d_conv`) in `model.layers.*.self_attn.k_conv1d` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 20축
- `d_head_kda` (← config `linear_attn_config`) in `model.layers.*.self_attn.k_conv1d` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 20축
- `n_h_kda` (← config `linear_attn_config`) in `model.layers.*.self_attn.k_conv1d` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 20축
- `d_conv` (← config `conv_L_cache/conv_kernel/d_conv/linear_attn_config/mamba_d_conv`) in `model.layers.*.self_attn.k_conv1d.conv` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 20축
- `d_head_kda` (← config `linear_attn_config`) in `model.layers.*.self_attn.k_conv1d.conv` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 20축
- `n_h_kda` (← config `linear_attn_config`) in `model.layers.*.self_attn.k_conv1d.conv` — 이 모듈도 이 모듈을 만든 `model / (root)` 도 그 필드를 읽지 않는다, 20축

### 0. 규칙이 끝내지 못한 축 — **여기부터 답한다**

값으로는 결정할 수 없어 파이프라인이 판단을 넘긴 자리다. 세 가지뿐이다:
`tie`(두 심볼이 같은 값이라 관례로 골랐다) · `heur`(등록 규칙이 없어 산술로 지어냈다) · `bare`(이름을 못 붙였는데 크기가 커서 진짜 차원일 수 있다).

**답이 나오면 `override_stub` 을 채워 `rules/label_overrides.yaml` 에 넣는다.** `spread: class` 라 그 축이 지나는 모든 자리가 한 번에 바뀐다 — 모듈 경계에서 멈추지 않는다(그것이 예전에 교정을 막던 유일한 이유였다).

**값이 같은 심볼이 여럿이면 값으로는 영원히 못 가른다. shape 안의 위치가 말해 준다** — `[B, n_h, T, d_head]` 의 축 1 은 head 개수, 축 3 은 head 폭이다.

아래 `shape` 과 `축` 은 **그 축을 처음 만든 자리(앵커)** 의 것이다. 초안은 `shape`/`axis`/`field`/`shape_index`/`op_type`/`nth` 여섯으로 그 앵커를 지목한다 — `shape`+`axis` 만으로는 부족하다(Kimi 의 `[B, n_h, T, d_nope]` 축 3 은 **366개 등가류**에 걸쳐 있다: q 의 q_pass, KV 의 k_nope, value_states …). `nth` 는 그 모듈 안에서 같은 op_type 의 몇 번째인지다 — MLA 는 `self_attn` 안에 `split_with_sizes` 가 q용·kv용 둘이라 그것 없이는 못 가른다.

**유일성은 실제로 돌려 봐서 검증한다**: 그 조건에 맞는 자리들이 몇 개의 등가류에 속하는지 세고, **한 레이어 안에서 둘 이상**이면 `stub_ambiguous` 를 붙인다. 그 초안은 쓰지 말고 `open` 으로 남길 것.

| 왜 | 모듈 | 크기 | 지금 이름 | 후보 | 축 | 앵커 shape | 축 수 |
|---|---|---|---|---|---|---|---|
| `tie` | `model.layers.*.self_attn` | 32 | `n_h_kda` | `n_h`, `n_kv` | 1 | `[B, n_h_kda, d_head_kda, d_head_kda]` | 3220 |
| `tie` | `model.layers.*.self_attn` | 128 | `d_head_kda` | `d_nope`, `d_v` | 2 | `[B, n_h_kda, d_head_kda, d_head_kda]` | 3220 |
| `tie` | `model.layers.*.self_attn` | 128 | `d_head_kda` | `d_nope`, `d_v` | 3 | `[B, n_h_kda, d_head_kda, d_head_kda]` | 3220 |
| `tie` | `model.layers.*.self_attn` | 32 | `n_h_kda` | `n_h`, `n_kv` | 2 | `[B, T, n_h_kda, d_head_kda]` | 1140 |
| `tie` | `model.layers.*.self_attn` | 128 | `d_head_kda` | `d_nope`, `d_v` | 3 | `[B, T, n_h_kda, d_head_kda]` | 1140 |
| `tie` | `model.layers.*.self_attn` | 32 | `n_h_kda` | `n_h`, `n_kv` | 2 | `[B, 1, n_h_kda, d_head_kda]` | 540 |
| `tie` | `model.layers.*.self_attn` | 128 | `d_head_kda` | `d_nope`, `d_v` | 3 | `[B, 1, n_h_kda, d_head_kda]` | 540 |
| `tie` | `model.layers.*.self_attn.b_proj` | 32 | `n_h_kda` | `n_h`, `n_kv` | 1 | `[d_model, n_h_kda]` | 500 |
| `heur` | `model.layers.*.block_sparse_moe` | 32 | `4*k` | — | 0 | `[4*k, d_model]` | 312 |
| `tie` | `model.layers.*.self_attn` | 32 | `n_h_kda` | `n_h`, `n_kv` | 1 | `[B, n_h_kda, d_head_kda]` | 180 |
| `tie` | `model.layers.*.self_attn` | 128 | `d_head_kda` | `d_nope`, `d_v` | 2 | `[B, n_h_kda, d_head_kda]` | 180 |
| `tie` | `model.layers.*.self_attn` | 128 | `d_head_kda` | `d_nope`, `d_v` | 3 | `[n_h_kda, d_head_kda, B, d_head_kda]` | 160 |
| `tie` | `model.layers.*.self_attn` | 32 | `n_h_kda` | `n_h`, `n_kv` | 2 | `[1, 1, n_h_kda, 1]` | 140 |
| `tie` | `model.layers.*.self_attn` | 32 | `n_h` | `n_h`, `n_kv` | 0 | `[n_h, T, T]` | 126 |
| `tie` | `model.layers.*.self_attn` | 32 | `n_h` | `n_h`, `n_kv` | 0 | `[n_h, B, T+1]` | 126 |
| `tie` | `model.layers.*.self_attn` | 128 | `d_nope` | `d_nope`, `d_v` | 3 | `[B, n_h, T, d_nope]` | 84 |
| `tie` | `model.layers.*.self_attn` | 32 | `n_h_kda` | `n_h`, `n_kv` | 1 | `[B, n_h_kda]` | 80 |
| `tie` | `model.layers.*.self_attn` | 128 | `d_head_kda` | `d_nope`, `d_v` | 2 | `[n_h_kda, B, d_head_kda]` | 80 |
| `tie` | `model.layers.*.self_attn` | 32 | `n_h` | `n_h`, `n_kv` | 1 | `[B, n_h, T, d_nope]` | 70 |
| `tie` | `model.layers.*.self_attn` | 32 | `n_h` | `n_h`, `n_kv` | 1 | `[B, n_h, 1, d_nope]` | 70 |
| `tie` | `model.layers.*.self_attn` | 32 | `n_h` | `n_h`, `n_kv` | 1 | `[B, n_h, 1, d_rope]` | 56 |
| `tie` | `model.layers.*.self_attn` | 32 | `n_h` | `n_h`, `n_kv` | 1 | `[B, n_h, d_nope+d_rope, T]` | 42 |
| `tie` | `model.layers.*.self_attn` | 32 | `n_h` | `n_h`, `n_kv` | 1 | `[B, n_h, d_nope+d_rope, T+1]` | 42 |
| `tie` | `model.layers.*.self_attn` | 32 | `n_h_kda` | `n_h`, `n_kv` | 0 | `[n_h_kda, d_head_kda]` | 40 |
| `tie` | `model.layers.*.self_attn` | 128 | `d_head_kda` | `d_nope`, `d_v` | 1 | `[n_h_kda, d_head_kda]` | 40 |
| `tie` | `model.layers.*.self_attn` | 32 | `n_h_kda` | `n_h`, `n_kv` | 2 | `[B, T, n_h_kda, 1]` | 40 |
| `tie` | `model.layers.*.self_attn` | 32 | `n_h_kda` | `n_h`, `n_kv` | 1 | `[B, n_h_kda, d_head_kda, 1]` | 40 |
| `tie` | `model.layers.*.self_attn` | 128 | `d_head_kda` | `d_nope`, `d_v` | 2 | `[B, n_h_kda, d_head_kda, 1]` | 40 |
| `tie` | `model.layers.*.self_attn` | 32 | `n_h_kda` | `n_h`, `n_kv` | 1 | `[B, n_h_kda, 1, d_head_kda]` | 40 |
| `tie` | `model.layers.*.self_attn` | 128 | `d_head_kda` | `d_nope`, `d_v` | 3 | `[B, n_h_kda, 1, d_head_kda]` | 40 |
| `tie` | `model.layers.*.self_attn` | 32 | `n_h_kda` | `n_h`, `n_kv` | 0 | `[n_h_kda, d_head_kda, B, 1]` | 40 |
| `tie` | `model.layers.*.self_attn` | 128 | `d_head_kda` | `d_nope`, `d_v` | 1 | `[n_h_kda, d_head_kda, B, 1]` | 40 |
| `tie` | `model.layers.*.self_attn` | 32 | `n_h_kda` | `n_h`, `n_kv` | 0 | `[n_h_kda, B, d_head_kda]` | 40 |
| `tie` | `model.layers.*.self_attn` | 32 | `n_h_kda` | `n_h`, `n_kv` | 0 | `[n_h_kda, d_head_kda, B, d_head_kda]` | 40 |
| `tie` | `model.layers.*.self_attn` | 128 | `d_head_kda` | `d_nope`, `d_v` | 1 | `[n_h_kda, d_head_kda, B, d_head_kda]` | 40 |
| `tie` | `model.layers.*.self_attn` | 32 | `n_h_kda` | `n_h`, `n_kv` | 0 | `[n_h_kda, d_head_kda, d_head_kda]` | 40 |
| `tie` | `model.layers.*.self_attn` | 32 | `n_h_kda` | `n_h`, `n_kv` | 0 | `[n_h_kda, B, 1, d_head_kda]` | 40 |
| `tie` | `model.layers.*.self_attn` | 32 | `n_h_kda` | `n_h`, `n_kv` | 2 | `[B, 1, n_h_kda, 1]` | 40 |
| `tie` | `model.layers.*.self_attn` | 32 | `n_h` | `n_h`, `n_kv` | 0 | `[n_h, B, d_nope]` | 28 |
| `tie` | `model.layers.*.self_attn` | 32 | `n_h` | `n_h`, `n_kv` | 2 | `[B, 1, n_h, d_nope+d_rope]` | 14 |

**고칠 것과 맞는 것 둘 다 적는다.** 이름이 틀렸으면 아래 초안의 `to`/`source` 를 채워 `rules/label_overrides.yaml` 에, **지금 이름이 맞으면** 같은 앵커에 `to` 대신 `label: <지금 이름>` 과 `source` 를 적어 `rules/label_confirmed.yaml` 에 넣는다. 확인을 적지 않으면 그 축은 재생성마다 다시 질문으로 올라온다.

초안(그대로 복사해 `to` 와 `source` 만 채운다):

```yaml
  - model: moonshotai__Kimi-Linear-48B-A3B-Instruct
    module: 'self_attn$'
    spread: class
    shape: ["B", "n_h_kda", "d_head_kda", "d_head_kda"]
    axis: 1
    field: o
    shape_index: 0
    op_type: new_zeros
    nth: 0
    from: n_h_kda
    to: <소스가 말하는 이름>
    expect: 32
    source: <modeling_*.py:줄 인용>
  - model: moonshotai__Kimi-Linear-48B-A3B-Instruct
    module: 'self_attn$'
    spread: class
    shape: ["B", "n_h_kda", "d_head_kda", "d_head_kda"]
    axis: 2
    field: o
    shape_index: 0
    op_type: new_zeros
    nth: 0
    from: d_head_kda
    to: <소스가 말하는 이름>
    expect: 128
    source: <modeling_*.py:줄 인용>
  - model: moonshotai__Kimi-Linear-48B-A3B-Instruct
    module: 'self_attn$'
    spread: class
    shape: ["B", "n_h_kda", "d_head_kda", "d_head_kda"]
    axis: 3
    field: o
    shape_index: 0
    op_type: new_zeros
    nth: 0
    from: d_head_kda
    to: <소스가 말하는 이름>
    expect: 128
    source: <modeling_*.py:줄 인용>
  - model: moonshotai__Kimi-Linear-48B-A3B-Instruct
    module: 'self_attn$'
    spread: class
    shape: ["B", "T", "n_h_kda", "d_head_kda"]
    axis: 2
    field: o
    shape_index: 0
    op_type: view
    nth: 5
    from: n_h_kda
    to: <소스가 말하는 이름>
    expect: 32
    source: <modeling_*.py:줄 인용>
  - model: moonshotai__Kimi-Linear-48B-A3B-Instruct
    module: 'self_attn$'
    spread: class
    shape: ["B", "T", "n_h_kda", "d_head_kda"]
    axis: 3
    field: o
    shape_index: 0
    op_type: view
    nth: 5
    from: d_head_kda
    to: <소스가 말하는 이름>
    expect: 128
    source: <modeling_*.py:줄 인용>
  - model: moonshotai__Kimi-Linear-48B-A3B-Instruct
    module: 'self_attn$'
    spread: class
    shape: ["B", "1", "n_h_kda", "d_head_kda"]
    axis: 2
    field: o
    shape_index: 0
    op_type: view
    nth: 5
    from: n_h_kda
    to: <소스가 말하는 이름>
    expect: 32
    source: <modeling_*.py:줄 인용>
```

## 기계적으로 이미 확인된 것 — 다시 묻지 말 것

- **심볼이 읽은 config 필드**: 3건이 클래스 선언 밖 — 위 1절 참고
- **정사각 축**: 소스에서 정사각 생성/reshape 과 대응이 확인된 축 없음
- **모듈이 읽는 config 속성**: `__init__` 에서 config 를 읽는 클래스 10개를 소스에서 확인했다. 그 목록이 각 모듈의 폭이 가질 수 있는 이름의 전부다.

## 행 단위 전건 — 여기부터 읽는다

접힌 표(`<phase>.jsonl`)의 **고유 행 전부**다. 레이어 인덱스만 접었고 그 밖에는 아무것도 합치지 않았다. 아래 A/B/C 절은 (모듈, 라벨)로 접은 뷰라 **한 행 안의 어긋남이 보이지 않는다** — 실제로 외부 검토가 찾아낸 결함 세 건이 전부 그 자리에 있었다.

**한 행씩 읽고 이것만 물어라: 입력·가중치·출력이 서로 말이 되는가.**

- 같은 텐서가 `input_shape` 와 `weight_shape` 에서 다른 이름을 쓰지 않는가 (가중치는 `[out, in]` 으로 저장되고 피연산자는 전치돼 있다)
- 전치·view 처럼 **이름을 바꿀 수 없는 op** 이 이름을 바꾸지 않았는가
- 행렬곱의 수축 축이 양쪽에서 같은 이름인가 — `[m,k] @ [k,n] -> [m,n]`
- 이 모듈이 그 이름을 가질 수 있는가 (소스에서 그 `nn.Linear` 를 만드는 줄을 찾아라)

고유 행 108개.

| phase | 모듈 | op | input_shape | weight_shape | output_shape |
|---|---|---|---|---|---|
| prefill | `model.embed_tokens` | embedding | `[['V', 'd_model'], ['B', 'T']]` | `['V', 'd_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.input_layernorm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.self_attn.q_proj` | matmul | `[['T', 'd_model'], ['d_model', 'n_h_kda*d_head_kda']]` | `['n_h_kda*d_head_kda', 'd_model']` | `[['T', 'n_h_kda*d_head_kda']]` |
| prefill | `model.layers.*.self_attn.q_conv1d.conv` | conv1d | `[['B', 'n_h_kda*d_head_kda', 'T'], ['n_h_kda*d_head_kda', '1', 'd_conv']]` | `['n_h_kda*d_head_kda', '1', 'd_conv']` | `[['B', 'n_h_kda*d_head_kda', 'T+d_conv-1']]` |
| prefill | `model.layers.*.self_attn.q_conv1d` | silu | `[['B', 'T', 'n_h_kda*d_head_kda']]` | `None` | `[['B', 'T', 'n_h_kda*d_head_kda']]` |
| prefill | `model.layers.*.self_attn.k_proj` | matmul | `[['T', 'd_model'], ['d_model', 'n_h_kda*d_head_kda']]` | `['n_h_kda*d_head_kda', 'd_model']` | `[['T', 'n_h_kda*d_head_kda']]` |
| prefill | `model.layers.*.self_attn.k_conv1d.conv` | conv1d | `[['B', 'n_h_kda*d_head_kda', 'T'], ['n_h_kda*d_head_kda', '1', 'd_conv']]` | `['n_h_kda*d_head_kda', '1', 'd_conv']` | `[['B', 'n_h_kda*d_head_kda', 'T+d_conv-1']]` |
| prefill | `model.layers.*.self_attn.k_conv1d` | silu | `[['B', 'T', 'n_h_kda*d_head_kda']]` | `None` | `[['B', 'T', 'n_h_kda*d_head_kda']]` |
| prefill | `model.layers.*.self_attn.v_proj` | matmul | `[['T', 'd_model'], ['d_model', 'n_h_kda*d_head_kda']]` | `['n_h_kda*d_head_kda', 'd_model']` | `[['T', 'n_h_kda*d_head_kda']]` |
| prefill | `model.layers.*.self_attn.v_conv1d.conv` | conv1d | `[['B', 'n_h_kda*d_head_kda', 'T'], ['n_h_kda*d_head_kda', '1', 'd_conv']]` | `['n_h_kda*d_head_kda', '1', 'd_conv']` | `[['B', 'n_h_kda*d_head_kda', 'T+d_conv-1']]` |
| prefill | `model.layers.*.self_attn.v_conv1d` | silu | `[['B', 'T', 'n_h_kda*d_head_kda']]` | `None` | `[['B', 'T', 'n_h_kda*d_head_kda']]` |
| prefill | `model.layers.*.self_attn.f_a_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_head_kda']]` | `['d_head_kda', 'd_model']` | `[['T', 'd_head_kda']]` |
| prefill | `model.layers.*.self_attn.f_b_proj` | matmul | `[['T', 'd_head_kda'], ['d_head_kda', 'n_h_kda*d_head_kda']]` | `['n_h_kda*d_head_kda', 'd_head_kda']` | `[['T', 'n_h_kda*d_head_kda']]` |
| prefill | `model.layers.*.self_attn` | exp | `[['n_h_kda', '1']]` | `['1', '1', 'n_h_kda', '1']` | `[['n_h_kda', '1']]` |
| prefill | `model.layers.*.self_attn.b_proj` | matmul | `[['T', 'd_model'], ['d_model', 'n_h_kda']]` | `['n_h_kda', 'd_model']` | `[['T', 'n_h_kda']]` |
| prefill | `model.layers.*.self_attn` | sigmoid | `[['B', 'T', 'n_h_kda']]` | `None` | `[['B', 'T', 'n_h_kda']]` |
| prefill | `model.layers.*.self_attn` | exp | `[['B', 'n_h_kda', 'd_head_kda', '1']]` | `None` | `[['B', 'n_h_kda', 'd_head_kda', '1']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['n_h_kda', 'B', 'd_head_kda'], ['n_h_kda', 'd_head_kda', 'd_head_kda']]` | `None` | `[['n_h_kda', 'B', 'd_head_kda']]` |
| prefill | `model.layers.*.self_attn.g_a_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_head_kda']]` | `['d_head_kda', 'd_model']` | `[['T', 'd_head_kda']]` |
| prefill | `model.layers.*.self_attn.g_b_proj` | matmul | `[['T', 'd_head_kda'], ['d_head_kda', 'n_h_kda*d_head_kda']]` | `['n_h_kda*d_head_kda', 'd_head_kda']` | `[['T', 'n_h_kda*d_head_kda']]` |
| prefill | `model.layers.*.self_attn.o_norm` | rmsnorm | `[['B', 'T', 'n_h_kda', 'd_head_kda']]` | `['d_head_kda']` | `[['B', 'T', 'n_h_kda', 'd_head_kda']]` |
| prefill | `model.layers.*.self_attn.o_proj` | matmul | `[['T', 'n_h_kda*d_head_kda'], ['n_h_kda*d_head_kda', 'd_model']]` | `['d_model', 'n_h_kda*d_head_kda']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_add | `[['B', 'T', 'd_model'], ['B', 'T', 'd_model']]` | `None` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.post_attention_layernorm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.mlp.gate_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['T', 'd_ff']]` |
| prefill | `model.layers.*.mlp.act_fn` | silu | `[['B', 'T', 'd_ff']]` | `None` | `[['B', 'T', 'd_ff']]` |
| prefill | `model.layers.*.mlp.up_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['T', 'd_ff']]` |
| prefill | `model.layers.*.mlp` | elementwise_mul | `[['B', 'T', 'd_ff'], ['B', 'T', 'd_ff']]` | `None` | `[['B', 'T', 'd_ff']]` |
| prefill | `model.layers.*.mlp.down_proj` | matmul | `[['T', 'd_ff'], ['d_ff', 'd_model']]` | `['d_model', 'd_ff']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*.block_sparse_moe.gate` | matmul | `[['T', 'd_model'], ['d_model', 'E']]` | `['E', 'd_model']` | `[['T', 'E']]` |
| prefill | `model.layers.*.block_sparse_moe.gate` | sigmoid | `[['T', 'E']]` | `None` | `[['T', 'E']]` |
| prefill | `model.layers.*.block_sparse_moe.experts.*.w1` | matmul | `[['4*k', 'd_model'], ['d_model', 'd_moe']]` | `['d_moe', 'd_model']` | `[['4*k', 'd_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.experts.*.act_fn` | silu | `[['4*k', 'd_moe']]` | `None` | `[['4*k', 'd_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.experts.*.w3` | matmul | `[['4*k', 'd_model'], ['d_model', 'd_moe']]` | `['d_moe', 'd_model']` | `[['4*k', 'd_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.experts.*` | elementwise_mul | `[['4*k', 'd_moe'], ['4*k', 'd_moe']]` | `None` | `[['4*k', 'd_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.experts.*.w2` | matmul | `[['4*k', 'd_moe'], ['d_moe', 'd_model']]` | `['d_model', 'd_moe']` | `[['4*k', 'd_model']]` |
| prefill | `model.layers.*.block_sparse_moe` | concat | `[['4*k', 'd_model'], ['4*k', 'd_model'], ['4*k', 'd_model'], ['4*k', 'd_model']]` | `None` | `[['k*T', 'd_model']]` |
| prefill | `model.layers.*.block_sparse_moe` | sum | `[['T', 'k', 'd_model']]` | `None` | `[['T', 'd_model']]` |
| prefill | `model.layers.*.block_sparse_moe.shared_experts.gate_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_moe']]` | `['d_moe', 'd_model']` | `[['T', 'd_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.shared_experts.act_fn` | silu | `[['B', 'T', 'd_moe']]` | `None` | `[['B', 'T', 'd_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.shared_experts.up_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_moe']]` | `['d_moe', 'd_model']` | `[['T', 'd_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.shared_experts` | elementwise_mul | `[['B', 'T', 'd_moe'], ['B', 'T', 'd_moe']]` | `None` | `[['B', 'T', 'd_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.shared_experts.down_proj` | matmul | `[['T', 'd_moe'], ['d_moe', 'd_model']]` | `['d_model', 'd_moe']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*.block_sparse_moe` | elementwise_add | `[['B', 'T', 'd_model'], ['B', 'T', 'd_model']]` | `None` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.self_attn.q_proj` | matmul | `[['T', 'd_model'], ['d_model', 'n_h*(d_nope+d_rope)']]` | `['n_h*(d_nope+d_rope)', 'd_model']` | `[['T', 'n_h*(d_nope+d_rope)']]` |
| prefill | `model.layers.*.self_attn.kv_a_proj_with_mqa` | matmul | `[['T', 'd_model'], ['d_model', 'c_kv+d_rope']]` | `['c_kv+d_rope', 'd_model']` | `[['T', 'c_kv+d_rope']]` |
| prefill | `model.layers.*.self_attn.kv_a_layernorm` | rmsnorm | `[['B', 'T', 'c_kv']]` | `['c_kv']` | `[['B', 'T', 'c_kv']]` |
| prefill | `model.layers.*.self_attn.kv_b_proj` | matmul | `[['T', 'c_kv'], ['c_kv', 'n_h*(d_nope+d_v)']]` | `['n_h*(d_nope+d_v)', 'c_kv']` | `[['T', 'n_h*(d_nope+d_v)']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'T', 'd_nope+d_rope'], ['n_h', 'd_nope+d_rope', 'T']]` | `None` | `[['n_h', 'T', 'T']]` |
| prefill | `model.layers.*.self_attn` | softmax | `[['B', 'n_h', 'T', 'T']]` | `None` | `[['B', 'n_h', 'T', 'T']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'T', 'T'], ['n_h', 'T', 'd_nope']]` | `None` | `[['n_h', 'T', 'd_nope']]` |
| prefill | `model.layers.*.self_attn.o_proj` | matmul | `[['T', 'n_h*d_v'], ['n_h*d_v', 'd_model']]` | `['d_model', 'n_h*d_v']` | `[['T', 'd_model']]` |
| prefill | `model.norm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `lm_head` | matmul | `[['T', 'd_model'], ['d_model', 'V']]` | `['V', 'd_model']` | `[['T', 'V']]` |
| decode | `model.embed_tokens` | embedding | `[['V', 'd_model'], ['B', '1']]` | `['V', 'd_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.input_layernorm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.self_attn.q_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_h_kda*d_head_kda']]` | `['n_h_kda*d_head_kda', 'd_model']` | `[['B', 'n_h_kda*d_head_kda']]` |
| decode | `model.layers.*.self_attn.q_conv1d` | conv1d | `[['B', 'n_h_kda*d_head_kda', 'd_conv'], ['n_h_kda*d_head_kda', '1', 'd_conv']]` | `['n_h_kda*d_head_kda', '1', 'd_conv']` | `[['B', 'n_h_kda*d_head_kda', '1']]` |
| decode | `model.layers.*.self_attn.q_conv1d` | silu | `[['B', '1', 'n_h_kda*d_head_kda']]` | `None` | `[['B', '1', 'n_h_kda*d_head_kda']]` |
| decode | `model.layers.*.self_attn.k_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_h_kda*d_head_kda']]` | `['n_h_kda*d_head_kda', 'd_model']` | `[['B', 'n_h_kda*d_head_kda']]` |
| decode | `model.layers.*.self_attn.k_conv1d` | conv1d | `[['B', 'n_h_kda*d_head_kda', 'd_conv'], ['n_h_kda*d_head_kda', '1', 'd_conv']]` | `['n_h_kda*d_head_kda', '1', 'd_conv']` | `[['B', 'n_h_kda*d_head_kda', '1']]` |
| decode | `model.layers.*.self_attn.k_conv1d` | silu | `[['B', '1', 'n_h_kda*d_head_kda']]` | `None` | `[['B', '1', 'n_h_kda*d_head_kda']]` |
| decode | `model.layers.*.self_attn.v_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_h_kda*d_head_kda']]` | `['n_h_kda*d_head_kda', 'd_model']` | `[['B', 'n_h_kda*d_head_kda']]` |
| decode | `model.layers.*.self_attn.v_conv1d` | conv1d | `[['B', 'n_h_kda*d_head_kda', 'd_conv'], ['n_h_kda*d_head_kda', '1', 'd_conv']]` | `['n_h_kda*d_head_kda', '1', 'd_conv']` | `[['B', 'n_h_kda*d_head_kda', '1']]` |
| decode | `model.layers.*.self_attn.v_conv1d` | silu | `[['B', '1', 'n_h_kda*d_head_kda']]` | `None` | `[['B', '1', 'n_h_kda*d_head_kda']]` |
| decode | `model.layers.*.self_attn.f_a_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_head_kda']]` | `['d_head_kda', 'd_model']` | `[['B', 'd_head_kda']]` |
| decode | `model.layers.*.self_attn.f_b_proj` | matmul | `[['B', 'd_head_kda'], ['d_head_kda', 'n_h_kda*d_head_kda']]` | `['n_h_kda*d_head_kda', 'd_head_kda']` | `[['B', 'n_h_kda*d_head_kda']]` |
| decode | `model.layers.*.self_attn` | exp | `[['n_h_kda', '1']]` | `['1', '1', 'n_h_kda', '1']` | `[['n_h_kda', '1']]` |
| decode | `model.layers.*.self_attn.b_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_h_kda']]` | `['n_h_kda', 'd_model']` | `[['B', 'n_h_kda']]` |
| decode | `model.layers.*.self_attn` | sigmoid | `[['B', '1', 'n_h_kda']]` | `None` | `[['B', '1', 'n_h_kda']]` |
| decode | `model.layers.*.self_attn` | exp | `[['B', 'n_h_kda', 'd_head_kda', '1']]` | `None` | `[['B', 'n_h_kda', 'd_head_kda', '1']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['n_h_kda', 'B', 'd_head_kda'], ['n_h_kda', 'd_head_kda', 'd_head_kda']]` | `None` | `[['n_h_kda', 'B', 'd_head_kda']]` |
| decode | `model.layers.*.self_attn.g_a_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_head_kda']]` | `['d_head_kda', 'd_model']` | `[['B', 'd_head_kda']]` |
| decode | `model.layers.*.self_attn.g_b_proj` | matmul | `[['B', 'd_head_kda'], ['d_head_kda', 'n_h_kda*d_head_kda']]` | `['n_h_kda*d_head_kda', 'd_head_kda']` | `[['B', 'n_h_kda*d_head_kda']]` |
| decode | `model.layers.*.self_attn.o_norm` | rmsnorm | `[['B', '1', 'n_h_kda', 'd_head_kda']]` | `['d_head_kda']` | `[['B', '1', 'n_h_kda', 'd_head_kda']]` |
| decode | `model.layers.*.self_attn.o_proj` | matmul | `[['B', 'n_h_kda*d_head_kda'], ['n_h_kda*d_head_kda', 'd_model']]` | `['d_model', 'n_h_kda*d_head_kda']` | `[['B', 'd_model']]` |
| decode | `model.layers.*` | elementwise_add | `[['B', '1', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.post_attention_layernorm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.mlp.gate_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['B', 'd_ff']]` |
| decode | `model.layers.*.mlp.act_fn` | silu | `[['B', '1', 'd_ff']]` | `None` | `[['B', '1', 'd_ff']]` |
| decode | `model.layers.*.mlp.up_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['B', 'd_ff']]` |
| decode | `model.layers.*.mlp` | elementwise_mul | `[['B', '1', 'd_ff'], ['B', '1', 'd_ff']]` | `None` | `[['B', '1', 'd_ff']]` |
| decode | `model.layers.*.mlp.down_proj` | matmul | `[['B', 'd_ff'], ['d_ff', 'd_model']]` | `['d_model', 'd_ff']` | `[['B', 'd_model']]` |
| decode | `model.layers.*.block_sparse_moe.gate` | matmul | `[['B', 'd_model'], ['d_model', 'E']]` | `['E', 'd_model']` | `[['B', 'E']]` |
| decode | `model.layers.*.block_sparse_moe.gate` | sigmoid | `[['B', 'E']]` | `None` | `[['B', 'E']]` |
| decode | `model.layers.*.block_sparse_moe.experts.*.w1` | matmul | `[['2', 'd_model'], ['d_model', 'd_moe']]` | `['d_moe', 'd_model']` | `[['2', 'd_moe']]` |
| decode | `model.layers.*.block_sparse_moe.experts.*.act_fn` | silu | `[['2', 'd_moe']]` | `None` | `[['2', 'd_moe']]` |
| decode | `model.layers.*.block_sparse_moe.experts.*.w3` | matmul | `[['2', 'd_model'], ['d_model', 'd_moe']]` | `['d_moe', 'd_model']` | `[['2', 'd_moe']]` |
| decode | `model.layers.*.block_sparse_moe.experts.*` | elementwise_mul | `[['2', 'd_moe'], ['2', 'd_moe']]` | `None` | `[['2', 'd_moe']]` |
| decode | `model.layers.*.block_sparse_moe.experts.*.w2` | matmul | `[['2', 'd_moe'], ['d_moe', 'd_model']]` | `['d_model', 'd_moe']` | `[['2', 'd_model']]` |
| decode | `model.layers.*.block_sparse_moe` | concat | `[['2', 'd_model'], ['2', 'd_model'], ['2', 'd_model'], ['2', 'd_model']]` | `None` | `[['k', 'd_model']]` |
| decode | `model.layers.*.block_sparse_moe` | sum | `[['B', 'k', 'd_model']]` | `None` | `[['B', 'd_model']]` |
| decode | `model.layers.*.block_sparse_moe.shared_experts.gate_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_moe']]` | `['d_moe', 'd_model']` | `[['B', 'd_moe']]` |
| decode | `model.layers.*.block_sparse_moe.shared_experts.act_fn` | silu | `[['B', '1', 'd_moe']]` | `None` | `[['B', '1', 'd_moe']]` |
| decode | `model.layers.*.block_sparse_moe.shared_experts.up_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_moe']]` | `['d_moe', 'd_model']` | `[['B', 'd_moe']]` |
| decode | `model.layers.*.block_sparse_moe.shared_experts` | elementwise_mul | `[['B', '1', 'd_moe'], ['B', '1', 'd_moe']]` | `None` | `[['B', '1', 'd_moe']]` |
| decode | `model.layers.*.block_sparse_moe.shared_experts.down_proj` | matmul | `[['B', 'd_moe'], ['d_moe', 'd_model']]` | `['d_model', 'd_moe']` | `[['B', 'd_model']]` |
| decode | `model.layers.*.block_sparse_moe` | elementwise_add | `[['B', '1', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.self_attn.q_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_h*(d_nope+d_rope)']]` | `['n_h*(d_nope+d_rope)', 'd_model']` | `[['B', 'n_h*(d_nope+d_rope)']]` |
| decode | `model.layers.*.self_attn.kv_a_proj_with_mqa` | matmul | `[['B', 'd_model'], ['d_model', 'c_kv+d_rope']]` | `['c_kv+d_rope', 'd_model']` | `[['B', 'c_kv+d_rope']]` |
| decode | `model.layers.*.self_attn.kv_a_layernorm` | rmsnorm | `[['B', '1', 'c_kv']]` | `['c_kv']` | `[['B', '1', 'c_kv']]` |
| decode | `model.layers.*.self_attn.kv_b_proj` | matmul | `[['B', 'c_kv'], ['c_kv', 'n_h*(d_nope+d_v)']]` | `['n_h*(d_nope+d_v)', 'c_kv']` | `[['B', 'n_h*(d_nope+d_v)']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'B', 'd_nope+d_rope'], ['n_h', 'd_nope+d_rope', 'T+1']]` | `None` | `[['n_h', 'B', 'T+1']]` |
| decode | `model.layers.*.self_attn` | softmax | `[['B', 'n_h', '1', 'T+1']]` | `None` | `[['B', 'n_h', '1', 'T+1']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'B', 'T+1'], ['n_h', 'T+1', 'd_nope']]` | `None` | `[['n_h', 'B', 'd_nope']]` |
| decode | `model.layers.*.self_attn.o_proj` | matmul | `[['B', 'n_h*d_v'], ['n_h*d_v', 'd_model']]` | `['d_model', 'n_h*d_v']` | `[['B', 'd_model']]` |
| decode | `model.norm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `lm_head` | matmul | `[['B', 'd_model'], ['d_model', 'V']]` | `['V', 'd_model']` | `[['B', 'V']]` |

## 전수 점검 — 이 모델이 쓰는 이름 전부

위 절이 '풀리지 않은 것'이라면 여기는 **전부**다. 규칙이 자신 있게 붙인 이름도 틀릴 수 있고, 그런 건 미결 목록에 절대 오르지 않는다. 한 줄씩 읽고 **그 모듈에서 그 이름이 말이 되는지** 보라.

### A. 붙은 이름 전부 (27종)

| 라벨 | 값 | 나타나는 모듈 | 축 수 |
|---|---|---|---|
| `B` |  | `model.layers.*.self_attn`, `model.layers.*.block_sparse_moe.gate`, `model.layers.*.input_layernorm`, `model.layers.*.post_attention_layernorm` 외 61개 | 38331 |
| `d_head_kda` | 128 | `model.layers.*.self_attn`, `model.layers.*.self_attn.o_norm`, `model.layers.*.self_attn.f_a_proj`, `model.layers.*.self_attn.f_b_proj` 외 2개 | 33260 |
| `n_h_kda` | 32 | `model.layers.*.self_attn`, `model.layers.*.self_attn.o_norm`, `model.layers.*.self_attn.b_proj` | 29500 |
| `d_model` | 2304 | `model.layers.*.block_sparse_moe`, `model.layers.*.block_sparse_moe.experts.*.w1`, `model.layers.*.block_sparse_moe.experts.*.w3`, `model.layers.*.block_sparse_moe.experts.*.w2` 외 47개 | 11882 |
| `T` |  | `model.layers.*.self_attn`, `model.layers.*.block_sparse_moe.gate`, `model.layers.*.block_sparse_moe`, `model.layers.*.input_layernorm` 외 61개 | 10229 |
| `d_moe` | 1024 | `model.layers.*.block_sparse_moe.experts.*.w1`, `model.layers.*.block_sparse_moe.experts.*.w3`, `model.layers.*.block_sparse_moe.experts.*.w2`, `model.layers.*.block_sparse_moe.experts.*.act_fn` 외 9개 | 6292 |
| `n_h_kda*d_head_kda` |  | `model.layers.*.self_attn.q_conv1d`, `model.layers.*.self_attn.k_conv1d`, `model.layers.*.self_attn.v_conv1d`, `model.layers.*.self_attn.q_proj` 외 9개 | 4060 |
| `E` | 256 | `model.layers.*.block_sparse_moe.gate`, `model.layers.*.block_sparse_moe` | 1872 |
| `k` | 8 | `model.layers.*.block_sparse_moe`, `model.layers.*.block_sparse_moe.gate` | 1612 |
| `4*k` |  | `model.layers.*.block_sparse_moe`, `model.layers.*.block_sparse_moe.experts.*.w1`, `model.layers.*.block_sparse_moe.experts.*.act_fn`, `model.layers.*.block_sparse_moe.experts.*.w3` 외 5개 | 1352 |
| `n_h` | 32 | `model.layers.*.self_attn` | 896 |
| `k*T` |  | `model.layers.*.block_sparse_moe` | 520 |
| `d_conv` | 4 | `model.layers.*.self_attn.q_conv1d`, `model.layers.*.self_attn.k_conv1d`, `model.layers.*.self_attn.v_conv1d`, `model.layers.*.self_attn.q_conv1d.conv` 외 2개 | 420 |
| `c_kv` | 512 | `model.layers.*.self_attn.kv_a_layernorm`, `model.layers.*.self_attn.kv_b_proj`, `model.layers.*.self_attn` | 294 |
| `d_nope+d_rope` |  | `model.layers.*.self_attn` | 273 |
| `T+1` |  | `model.layers.*.self_attn`, `model` | 260 |
| `d_nope` | 128 | `model.layers.*.self_attn` | 245 |
| `n_h*(d_nope+d_rope)` |  | `model.layers.*.self_attn.q_proj`, `model.layers.*.self_attn` | 126 |
| `c_kv+d_rope` |  | `model.layers.*.self_attn.kv_a_proj_with_mqa`, `model.layers.*.self_attn` | 126 |
| `n_h*(d_nope+d_v)` |  | `model.layers.*.self_attn.kv_b_proj`, `model.layers.*.self_attn` | 126 |
| `n_h*d_v` |  | `model.layers.*.self_attn.o_proj`, `model.layers.*.self_attn` | 126 |
| `T+d_conv-1` |  | `model.layers.*.self_attn.q_conv1d.conv`, `model.layers.*.self_attn.q_conv1d`, `model.layers.*.self_attn.k_conv1d.conv`, `model.layers.*.self_attn.k_conv1d` 외 2개 | 120 |
| `d_rope` | 64 | `model.layers.*.self_attn` | 112 |
| `d_ff` | 9216 | `model.layers.*.mlp.gate_proj`, `model.layers.*.mlp.up_proj`, `model.layers.*.mlp.down_proj`, `model.layers.*.mlp` 외 1개 | 58 |
| `d_nope+d_v` |  | `model.layers.*.self_attn` | 56 |
| `V` | 163840 | `lm_head`, `model.embed_tokens` | 20 |
| `d_v` | 128 | `model.layers.*.self_attn` | 14 |

### B. 이름 없이 남은 정수 전부 (13쌍)

**여기가 필터가 못 보던 자리다.** 정수가 남는 것 자체는 정상이다(루프 인덱스, 피연산자 개수, 브로드캐스트 축). 문제는 **이름이 있어야 하는데 없는 경우**이고, 마지막 열이 그 신호다 — 이 모델의 심볼과 값이 같다면 스코프가 그 모듈을 못 덮고 있을 수 있다. 실제로 `n_hc`(=4)가 그렇게 정수로 남아 있었다.

| 모듈 | 정수 | 축 수 | 같은 값의 심볼 |
|---|---|---|---|
| `model.layers.*.block_sparse_moe` | 2 | 208 | — |
| `model.layers.*.block_sparse_moe.experts.*.w1` | 2 | 208 | — |
| `model.layers.*.block_sparse_moe.experts.*.act_fn` | 2 | 208 | — |
| `model.layers.*.block_sparse_moe.experts.*.w3` | 2 | 208 | — |
| `model.layers.*.block_sparse_moe.experts.*.w2` | 2 | 208 | — |
| `model.layers.*.block_sparse_moe.gate` | 2 | 156 | — |
| `model.layers.*.block_sparse_moe.experts.0` | 2 | 78 | — |
| `model.layers.*.block_sparse_moe.experts.1` | 2 | 78 | — |
| `model.layers.*.block_sparse_moe.experts.2` | 2 | 78 | — |
| `model.layers.*.block_sparse_moe.experts.3` | 2 | 78 | — |
| `model.layers.*.self_attn.q_conv1d` | 3 | 60 | — |
| `model.layers.*.self_attn.k_conv1d` | 3 | 60 | — |
| `model.layers.*.self_attn.v_conv1d` | 3 | 60 | — |

### C. 모듈이 내는 출력 shape 전부 (73개 모듈 / 376종)

모듈 하나가 어떤 모양을 내놓는지 전부 적었다. 어떤 모듈에 **있을 수 없는 이름**이 섞여 있는지 보는 자리다(예: attention head 수가 Mamba mixer 안에, 전문가 수가 self_attn 안에).

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
  - `[]`
- `model.embed_tokens`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.*.block_sparse_moe`
  - `[[2, d_model]]`
  - `[[4*k, d_model]]`
  - `[[B, 1, d_model]]`
  - `[[B, E]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[B, k, 1]]`
  - `[[B, k, d_model]]`
  - `[[T, E]]`
  - `[[T, d_model]]`
  - `[[T, k, 1]]`
  - `[[T, k, d_model]]`
  - `[[k*T, d_model]]`
  - `[[k*T], [k*T]]`
  - `[[k*T]]`
  - `[[k, d_model]]`
  - `[[k], [k]]`
  - `[[k]]`
- `model.layers.*.block_sparse_moe.experts.*.act_fn`
  - `[[2, d_moe]]`
  - `[[4*k, d_moe]]`
- `model.layers.*.block_sparse_moe.experts.*.w1`
  - `[[2, d_moe]]`
  - `[[4*k, d_moe]]`
  - `[[d_model, d_moe]]`
- `model.layers.*.block_sparse_moe.experts.*.w2`
  - `[[2, d_model]]`
  - `[[4*k, d_model]]`
  - `[[d_moe, d_model]]`
- `model.layers.*.block_sparse_moe.experts.*.w3`
  - `[[2, d_moe]]`
  - `[[4*k, d_moe]]`
  - `[[d_model, d_moe]]`
- `model.layers.*.block_sparse_moe.experts.0`
  - `[[2, d_moe]]`
  - `[[4*k, d_moe]]`
- `model.layers.*.block_sparse_moe.experts.1`
  - `[[2, d_moe]]`
  - `[[4*k, d_moe]]`
- `model.layers.*.block_sparse_moe.experts.2`
  - `[[2, d_moe]]`
  - `[[4*k, d_moe]]`
- `model.layers.*.block_sparse_moe.experts.3`
  - `[[2, d_moe]]`
  - `[[4*k, d_moe]]`
- `model.layers.*.block_sparse_moe.gate`
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
- `model.layers.*.block_sparse_moe.shared_experts`
  - `[[B, 1, d_moe]]`
  - `[[B, T, d_moe]]`
- `model.layers.*.block_sparse_moe.shared_experts.act_fn`
  - `[[B, 1, d_moe]]`
  - `[[B, T, d_moe]]`
- `model.layers.*.block_sparse_moe.shared_experts.down_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[B, d_moe]]`
  - `[[T, d_model]]`
  - `[[T, d_moe]]`
  - `[[d_moe, d_model]]`
- `model.layers.*.block_sparse_moe.shared_experts.gate_proj`
  - `[[B, 1, d_moe]]`
  - `[[B, T, d_moe]]`
  - `[[B, d_model]]`
  - `[[B, d_moe]]`
  - `[[T, d_model]]`
  - `[[T, d_moe]]`
  - `[[d_model, d_moe]]`
- `model.layers.*.block_sparse_moe.shared_experts.up_proj`
  - `[[B, 1, d_moe]]`
  - `[[B, T, d_moe]]`
  - `[[B, d_model]]`
  - `[[B, d_moe]]`
  - `[[T, d_model]]`
  - `[[T, d_moe]]`
  - `[[d_model, d_moe]]`
- `model.layers.*.input_layernorm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
- `model.layers.*.mlp`
  - `[[B, 1, d_ff]]`
  - `[[B, T, d_ff]]`
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
- `model.layers.*.mlp.gate_proj`
  - `[[B, 1, d_ff]]`
  - `[[B, T, d_ff]]`
  - `[[B, d_ff]]`
  - `[[B, d_model]]`
  - `[[T, d_ff]]`
  - `[[T, d_model]]`
  - `[[d_model, d_ff]]`
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
  - `[[B, 1, 1, T+1]]`
  - `[[B, 1, 1, d_rope]]`
  - `[[B, 1, T, T]]`
  - `[[B, 1, T, d_rope]]`
  - `[[B, 1, c_kv], [B, 1, d_rope]]`
  - `[[B, 1, n_h*d_v]]`
  - `[[B, 1, n_h, d_nope+d_rope]]`
  - `[[B, 1, n_h, d_nope+d_v]]`
  - `[[B, 1, n_h, d_nope]]`
  - `[[B, 1, n_h_kda*d_head_kda]]`
  - `[[B, 1, n_h_kda, 1, d_head_kda]]`
  - `[[B, 1, n_h_kda, 1]]`
  - `[[B, 1, n_h_kda, d_head_kda]]`
  - `[[B, 1, n_h_kda]]`
  - `[[B, T, c_kv], [B, T, d_rope]]`
  - `[[B, T, n_h*d_v]]`
  - `[[B, T, n_h, d_nope+d_rope]]`
  - `[[B, T, n_h, d_nope+d_v]]`
  - `[[B, T, n_h, d_nope]]`
  - `[[B, T, n_h_kda*d_head_kda]]`
  - `[[B, T, n_h_kda, 1, d_head_kda]]`
  - `[[B, T, n_h_kda, 1]]`
  - `[[B, T, n_h_kda, d_head_kda]]`
  - `[[B, T, n_h_kda]]`
  - `[[B, n_h, 1, T+1]]`
  - `[[B, n_h, 1, d_nope+d_rope]]`
  - `[[B, n_h, 1, d_nope+d_v]]`
  - `[[B, n_h, 1, d_nope], [B, n_h, 1, d_rope]]`
  - `[[B, n_h, 1, d_nope], [B, n_h, 1, d_v]]`
  - `[[B, n_h, 1, d_nope]]`
  - `[[B, n_h, 1, d_rope]]`
  - `[[B, n_h, T+1, d_nope+d_rope]]`
  - `[[B, n_h, T+1, d_nope]]`
  - `[[B, n_h, T, T]]`
  - `[[B, n_h, T, d_nope+d_rope]]`
  - `[[B, n_h, T, d_nope+d_v]]`
  - `[[B, n_h, T, d_nope], [B, n_h, T, d_rope]]`
  - `[[B, n_h, T, d_nope], [B, n_h, T, d_v]]`
  - `[[B, n_h, T, d_nope]]`
  - `[[B, n_h, T, d_rope]]`
  - `[[B, n_h, d_nope+d_rope, T+1]]`
  - `[[B, n_h, d_nope+d_rope, T]]`
  - `[[B, n_h_kda, 1, d_head_kda]]`
  - `[[B, n_h_kda, 1]]`
  - `[[B, n_h_kda, d_head_kda, 1]]`
  - `[[B, n_h_kda, d_head_kda, d_head_kda]]`
  - `[[B, n_h_kda, d_head_kda]]`
  - `[[B, n_h_kda]]`
  - `[[n_h, B, T+1]]`
  - `[[n_h, B, d_nope+d_rope]]`
  - `[[n_h, B, d_nope]]`
  - `[[n_h, T+1, d_nope]]`
  - `[[n_h, T, T]]`
  - `[[n_h, T, d_nope+d_rope]]`
  - `[[n_h, T, d_nope]]`
  - `[[n_h, d_nope+d_rope, T+1]]`
  - `[[n_h, d_nope+d_rope, T]]`
  - `[[n_h_kda, 1]]`
  - `[[n_h_kda, B, 1, d_head_kda]]`
  - `[[n_h_kda, B, d_head_kda]]`
  - `[[n_h_kda, d_head_kda, B, 1]]`
  - `[[n_h_kda, d_head_kda, B, d_head_kda]]`
  - `[[n_h_kda, d_head_kda, d_head_kda]]`
  - `[[n_h_kda, d_head_kda]]`
- `model.layers.*.self_attn.b_proj`
  - `[[B, 1, n_h_kda]]`
  - `[[B, T, n_h_kda]]`
  - `[[B, d_model]]`
  - `[[B, n_h_kda]]`
  - `[[T, d_model]]`
  - `[[T, n_h_kda]]`
  - `[[d_model, n_h_kda]]`
- `model.layers.*.self_attn.f_a_proj`
  - `[[B, 1, d_head_kda]]`
  - `[[B, T, d_head_kda]]`
  - `[[B, d_head_kda]]`
  - `[[B, d_model]]`
  - `[[T, d_head_kda]]`
  - `[[T, d_model]]`
  - `[[d_model, d_head_kda]]`
- `model.layers.*.self_attn.f_b_proj`
  - `[[B, 1, n_h_kda*d_head_kda]]`
  - `[[B, T, n_h_kda*d_head_kda]]`
  - `[[B, d_head_kda]]`
  - `[[B, n_h_kda*d_head_kda]]`
  - `[[T, d_head_kda]]`
  - `[[T, n_h_kda*d_head_kda]]`
  - `[[d_head_kda, n_h_kda*d_head_kda]]`
- `model.layers.*.self_attn.g_a_proj`
  - `[[B, 1, d_head_kda]]`
  - `[[B, T, d_head_kda]]`
  - `[[B, d_head_kda]]`
  - `[[B, d_model]]`
  - `[[T, d_head_kda]]`
  - `[[T, d_model]]`
  - `[[d_model, d_head_kda]]`
- `model.layers.*.self_attn.g_b_proj`
  - `[[B, 1, n_h_kda*d_head_kda]]`
  - `[[B, T, n_h_kda*d_head_kda]]`
  - `[[B, d_head_kda]]`
  - `[[B, n_h_kda*d_head_kda]]`
  - `[[T, d_head_kda]]`
  - `[[T, n_h_kda*d_head_kda]]`
  - `[[d_head_kda, n_h_kda*d_head_kda]]`
- `model.layers.*.self_attn.k_conv1d`
  - `[[B, 1, n_h_kda*d_head_kda]]`
  - `[[B, T, n_h_kda*d_head_kda]]`
  - `[[B, n_h_kda*d_head_kda, 1]]`
  - `[[B, n_h_kda*d_head_kda, 3]]`
  - `[[B, n_h_kda*d_head_kda, T]]`
  - `[[B, n_h_kda*d_head_kda, d_conv]]`
- `model.layers.*.self_attn.k_conv1d.conv`
  - `[[B, n_h_kda*d_head_kda, T+d_conv-1]]`
- `model.layers.*.self_attn.k_proj`
  - `[[B, 1, n_h_kda*d_head_kda]]`
  - `[[B, T, n_h_kda*d_head_kda]]`
  - `[[B, d_model]]`
  - `[[B, n_h_kda*d_head_kda]]`
  - `[[T, d_model]]`
  - `[[T, n_h_kda*d_head_kda]]`
  - `[[d_model, n_h_kda*d_head_kda]]`
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
- `model.layers.*.self_attn.o_norm`
  - `[[B, 1, n_h_kda, 1]]`
  - `[[B, 1, n_h_kda, d_head_kda]]`
  - `[[B, T, n_h_kda, 1]]`
  - `[[B, T, n_h_kda, d_head_kda]]`
- `model.layers.*.self_attn.o_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[B, n_h*d_v]]`
  - `[[B, n_h_kda*d_head_kda]]`
  - `[[T, d_model]]`
  - `[[T, n_h*d_v]]`
  - `[[T, n_h_kda*d_head_kda]]`
  - `[[n_h*d_v, d_model]]`
  - `[[n_h_kda*d_head_kda, d_model]]`
- `model.layers.*.self_attn.q_conv1d`
  - `[[B, 1, n_h_kda*d_head_kda]]`
  - `[[B, T, n_h_kda*d_head_kda]]`
  - `[[B, n_h_kda*d_head_kda, 1]]`
  - `[[B, n_h_kda*d_head_kda, 3]]`
  - `[[B, n_h_kda*d_head_kda, T]]`
  - `[[B, n_h_kda*d_head_kda, d_conv]]`
- `model.layers.*.self_attn.q_conv1d.conv`
  - `[[B, n_h_kda*d_head_kda, T+d_conv-1]]`
- `model.layers.*.self_attn.q_proj`
  - `[[B, 1, n_h*(d_nope+d_rope)]]`
  - `[[B, 1, n_h_kda*d_head_kda]]`
  - `[[B, T, n_h*(d_nope+d_rope)]]`
  - `[[B, T, n_h_kda*d_head_kda]]`
  - `[[B, d_model]]`
  - `[[B, n_h*(d_nope+d_rope)]]`
  - `[[B, n_h_kda*d_head_kda]]`
  - `[[T, d_model]]`
  - `[[T, n_h*(d_nope+d_rope)]]`
  - `[[T, n_h_kda*d_head_kda]]`
  - `[[d_model, n_h*(d_nope+d_rope)]]`
  - `[[d_model, n_h_kda*d_head_kda]]`
- `model.layers.*.self_attn.v_conv1d`
  - `[[B, 1, n_h_kda*d_head_kda]]`
  - `[[B, T, n_h_kda*d_head_kda]]`
  - `[[B, n_h_kda*d_head_kda, 1]]`
  - `[[B, n_h_kda*d_head_kda, 3]]`
  - `[[B, n_h_kda*d_head_kda, T]]`
  - `[[B, n_h_kda*d_head_kda, d_conv]]`
- `model.layers.*.self_attn.v_conv1d.conv`
  - `[[B, n_h_kda*d_head_kda, T+d_conv-1]]`
- `model.layers.*.self_attn.v_proj`
  - `[[B, 1, n_h_kda*d_head_kda]]`
  - `[[B, T, n_h_kda*d_head_kda]]`
  - `[[B, d_model]]`
  - `[[B, n_h_kda*d_head_kda]]`
  - `[[T, d_model]]`
  - `[[T, n_h_kda*d_head_kda]]`
  - `[[d_model, n_h_kda*d_head_kda]]`
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

## 이 의뢰서를 처리하는 법

`review/prompt.md` 를 LLM 에 넘기고 이 모델을 지정한다. 판정 4종과 근거 요건, 결과를 어디에 어떤 형식으로 쓰는지는 전부 `review/` 안에 있다.
