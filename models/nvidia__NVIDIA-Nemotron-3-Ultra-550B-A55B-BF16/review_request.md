# 검토 의뢰서 — nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `nemotron_h`
- 판단 필요: **3건**

## 증거 — 이미 받아둔 실제 소스

- `develop/sources/modeling_nemotron_h.py` — 있음, 이 파일을 열어서 판정한다
- `develop/sources/configuration_nemotron_h.py` — 있음, 이 파일을 열어서 판정한다

- 온라인 원본: https://github.com/huggingface/transformers/tree/main/src/transformers/models/nemotron_h

그 밖의 재료: `full/review.md`(리뷰 패킷 — shape 별 실제 행 표본), `structure.yaml`(이 모델의 심볼 표), `full/<phase>.csv`(전체 operator 표).

## 판단이 필요한 것

### 6. 값이 겹쳐 **임의로** 고른 축

두 심볼이 같은 값을 갖는 자리다. 규칙에는 고를 근거가 없고, 이긴 쪽은 전역 우선순위 — 즉 **관례**로 정해졌다. 이름이 맞을 수도 있지만 파이프라인은 그걸 알지 못한다. 표에서는 확신 있는 라벨과 똑같이 보인다.

**소스를 열어 어느 쪽인지 확정하는 것이 여기서 할 일이다.** 확정되면 `rules/label_overrides.yaml` 에 근거와 함께 못 박는다(review/05-overrides.md). 출신으로만 구별되는 경우라면 그렇게 적고 `open` 으로 남긴다.

- `d_chunk vs d_state` in `model.layers.*.mixer` — 값 128 를 두고 후보가 2개, 10800축
- `d_chunk vs d_head vs d_state` in `model.layers.*.mixer` — 값 128 를 두고 후보가 3개, 1296축
- `d_head_ssm vs n_h` in `model.layers.*.mixer` — 값 64 를 두고 후보가 2개, 1128축

### 0. 규칙이 끝내지 못한 축 — **여기부터 답한다**

값으로는 결정할 수 없어 파이프라인이 판단을 넘긴 자리다. 세 가지뿐이다:
`tie`(두 심볼이 같은 값이라 관례로 골랐다) · `heur`(등록 규칙이 없어 산술로 지어냈다) · `bare`(이름을 못 붙였는데 크기가 커서 진짜 차원일 수 있다).

**답이 나오면 `override_stub` 을 채워 `rules/label_overrides.yaml` 에 넣는다.** `spread: class` 라 그 축이 지나는 모든 자리가 한 번에 바뀐다 — 모듈 경계에서 멈추지 않는다(그것이 예전에 교정을 막던 유일한 이유였다).

**값이 같은 심볼이 여럿이면 값으로는 영원히 못 가른다. shape 안의 위치가 말해 준다** — `[B, n_h, T, d_head]` 의 축 1 은 head 개수, 축 3 은 head 폭이다.

아래 `shape` 과 `축` 은 **그 축을 처음 만든 자리(앵커)** 의 것이고, 초안의 `shape`/`axis` 가 그대로 그 앵커를 지목한다 — 같은 모듈에 같은 이름·같은 크기의 축이 여러 등가류로 나뉘어 있어도 서로를 훼손하지 않는다. 초안마다 유일성을 검증했다(`stub_ambiguous` 가 붙어 있으면 그 초안은 쓰지 말 것).

| 왜 | 모듈 | 크기 | 지금 이름 | 후보 | 축 | 앵커 shape | 축 수 |
|---|---|---|---|---|---|---|---|
| `tie` | `model.layers.*.mixer` | 64 | `d_head_ssm` | `d_head_ssm`, `n_h` | 2 | `[B, n_h_ssm, d_head_ssm, d_state]` | 1152 |
| `tie` | `model.layers.*.mixer` | 128 | `d_state` | `d_chunk`, `d_state` | 3 | `[B, n_h_ssm, d_head_ssm, d_state]` | 1152 |
| `tie` | `model.layers.*.mixer` | 128 | `d_state` | `d_chunk`, `d_state` | 1 | `[B, d_state, n_h_ssm, d_head_ssm]` | 1104 |
| `tie` | `model.layers.*.mixer` | 64 | `d_head_ssm` | `d_head_ssm`, `n_h` | 3 | `[B, d_state, n_h_ssm, d_head_ssm]` | 1104 |
| `tie` | `model.layers.*.mixer` | 128 | `d_chunk` | `d_chunk`, `d_state` | 3 | `[B, d_state, n_h_ssm, d_chunk]` | 864 |
| `tie` | `model.layers.*.mixer` | 64 | `d_head_ssm` | `d_head_ssm`, `n_h` | 2 | `[B, n_h_ssm, d_head_ssm]` | 864 |
| `tie` | `model.layers.*.mixer` | 128 | `d_state` | `d_chunk`, `d_state` | 1 | `[B, d_state, n_h_ssm, d_chunk]` | 672 |
| `tie` | `model.layers.*.mixer` | 128 | `d_state` | `d_chunk`, `d_state` | 3 | `[B, n_h_ssm, 1, d_state]` | 672 |
| `tie` | `model.layers.*.mixer` | 128 | `d_state` | `d_chunk`, `d_state` | 3 | `[B, n_g_ssm, n_h_ssm/n_g_ssm, d_state]` | 672 |
| `tie` | `model.layers.*.mixer` | 128 | `d_state` | `d_chunk`, `d_state` | 3 | `[B, T, n_g_ssm, d_state]` | 576 |
| `tie` | `model.layers.*.mixer` | 128 | `d_state` | `d_chunk`, `d_state` | 4 | `[B, T, n_g_ssm, n_h_ssm/n_g_ssm, d_state]` | 576 |
| `tie` | `model.layers.*.mixer` | 128 | `d_state` | `d_chunk`, `d_state` | 0 | `[d_state, d_chunk]` | 576 |
| `tie` | `model.layers.*.mixer` | 128 | `d_chunk` | `d_chunk`, `d_state` | 1 | `[d_state, d_chunk]` | 576 |
| `tie` | `model.layers.*.mixer` | 64 | `d_head_ssm` | `d_head_ssm`, `n_h` | 3 | `[B, 1, n_h_ssm, d_head_ssm, d_state]` | 528 |
| `tie` | `model.layers.*.mixer` | 128 | `d_state` | `d_chunk`, `d_state` | 4 | `[B, 1, n_h_ssm, d_head_ssm, d_state]` | 528 |
| `tie` | `model.layers.*.mixer` | 128 | `d_state` | `d_chunk`, `d_state` | 3 | `[B, n_h_ssm, 1, d_state, d_chunk]` | 480 |
| `tie` | `model.layers.*.mixer` | 128 | `d_chunk` | `d_chunk`, `d_state` | 4 | `[B, n_h_ssm, 1, d_state, d_chunk]` | 480 |
| `tie` | `model.layers.*.mixer` | 64 | `d_head_ssm` | `d_head_ssm`, `n_h` | 3 | `[B, T, n_h_ssm, d_head_ssm]` | 432 |
| `tie` | `model.layers.*.mixer` | 128 | `d_state` | `d_chunk`, `d_state` | 2 | `[B, 1, d_state, d_chunk, n_h_ssm, 1]` | 384 |
| `tie` | `model.layers.*.mixer` | 128 | `d_chunk` | `d_chunk`, `d_state` | 3 | `[B, 1, d_state, d_chunk, n_h_ssm, 1]` | 384 |
| `tie` | `model.layers.*.mixer` | 128 | `d_state` | `d_chunk`, `d_state` | 2 | `[B, n_g_ssm, d_state]` | 384 |
| `tie` | `model.layers.*.mixer` | 128 | `d_state` | `d_chunk`, `d_state` | 2 | `[B, 1, d_state, d_chunk, n_h_ssm]` | 288 |
| `tie` | `model.layers.*.mixer` | 128 | `d_chunk` | `d_chunk`, `d_state` | 3 | `[B, 1, d_state, d_chunk, n_h_ssm]` | 288 |
| `tie` | `model.layers.*.mixer` | 64 | `d_head_ssm` | `d_head_ssm`, `n_h` | 4 | `[B, 1, n_h_ssm, d_state, d_head_ssm]` | 288 |
| `tie` | `model.layers.*.mixer` | 64 | `d_head_ssm` | `d_head_ssm`, `n_h` | 3 | `[B, n_h_ssm, 2, d_head_ssm, d_state]` | 288 |
| `tie` | `model.layers.*.mixer` | 128 | `d_state` | `d_chunk`, `d_state` | 4 | `[B, n_h_ssm, 2, d_head_ssm, d_state]` | 288 |
| `tie` | `model.layers.*.mixer` | 64 | `d_head_ssm` | `d_head_ssm`, `n_h` | 2 | `[B, n_h_ssm, d_head_ssm, 1]` | 288 |
| `tie` | `model.layers.*.mixer` | 64 | `d_head_ssm` | `d_head_ssm`, `n_h` | 3 | `[B, 2, n_h_ssm, d_head_ssm, d_state]` | 240 |
| `tie` | `model.layers.*.mixer` | 128 | `d_state` | `d_chunk`, `d_state` | 4 | `[B, 2, n_h_ssm, d_head_ssm, d_state]` | 240 |
| `tie` | `model.layers.*.mixer` | 128 | `d_head` | `d_chunk`, `d_state` | 4 | `[B, 2, n_h_ssm/n_g_ssm, T, d_head]` | 240 |
| `tie` | `model.layers.*.mixer` | 128 | `d_head` | `d_chunk`, `d_state` | 4 | `[B, 2, n_h_ssm/n_g_ssm, T+1, d_head]` | 240 |
| `tie` | `model.layers.*.mixer` | 64 | `n_h` | `d_head_ssm`, `n_h` | 0 | `[n_h, T, T]` | 216 |
| `tie` | `model.layers.*.mixer` | 64 | `n_h` | `d_head_ssm`, `n_h` | 0 | `[n_h, B, T+1]` | 216 |
| `tie` | `model.layers.*.mixer` | 128 | `d_state` | `d_chunk`, `d_state` | 1 | `[B, d_state, n_h_ssm]` | 192 |
| `tie` | `model.layers.*.mixer` | 128 | `d_state` | `d_chunk`, `d_state` | 2 | `[B, 1, d_state, n_h_ssm]` | 192 |
| `tie` | `model.layers.*.mixer` | 128 | `d_state` | `d_chunk`, `d_state` | 2 | `[B, 1, d_state, n_h_ssm, 1]` | 192 |
| `tie` | `model.layers.*.mixer` | 128 | `d_state` | `d_chunk`, `d_state` | 3 | `[B, 1, n_h_ssm, d_state, d_head_ssm]` | 192 |
| `tie` | `model.layers.*.mixer` | 64 | `d_head_ssm` | `d_head_ssm`, `n_h` | 1 | `[n_h_ssm, d_head_ssm]` | 192 |
| `tie` | `model.layers.*.mixer` | 64 | `n_h` | `d_head_ssm`, `n_h` | 1 | `[B, n_h, T, d_head]` | 168 |
| `tie` | `model.layers.*.mixer` | 128 | `d_head` | `d_chunk`, `d_state` | 3 | `[B, n_h, T, d_head]` | 144 |

초안(그대로 복사해 `to` 와 `source` 만 채운다):

```yaml
  - model: nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
    module: 'mixer$'
    spread: class
    shape: ["B", "n_h_ssm", "d_head_ssm", "d_state"]
    axis: 2
    from: d_head_ssm
    to: <소스가 말하는 이름>
    expect: 64
    source: <modeling_*.py:줄 인용>
  - model: nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
    module: 'mixer$'
    spread: class
    shape: ["B", "n_h_ssm", "d_head_ssm", "d_state"]
    axis: 3
    from: d_state
    to: <소스가 말하는 이름>
    expect: 128
    source: <modeling_*.py:줄 인용>
  - model: nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
    module: 'mixer$'
    spread: class
    shape: ["B", "d_state", "n_h_ssm", "d_head_ssm"]
    axis: 1
    from: d_state
    to: <소스가 말하는 이름>
    expect: 128
    source: <modeling_*.py:줄 인용>
  - model: nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
    module: 'mixer$'
    spread: class
    shape: ["B", "d_state", "n_h_ssm", "d_head_ssm"]
    axis: 3
    from: d_head_ssm
    to: <소스가 말하는 이름>
    expect: 64
    source: <modeling_*.py:줄 인용>
  - model: nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
    module: 'mixer$'
    spread: class
    shape: ["B", "d_state", "n_h_ssm", "d_chunk"]
    axis: 3
    from: d_chunk
    to: <소스가 말하는 이름>
    expect: 128
    source: <modeling_*.py:줄 인용>
  - model: nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
    module: 'mixer$'
    spread: class
    shape: ["B", "n_h_ssm", "d_head_ssm"]
    axis: 2
    from: d_head_ssm
    to: <소스가 말하는 이름>
    expect: 64
    source: <modeling_*.py:줄 인용>
```

## 기계적으로 이미 확인된 것 — 다시 묻지 말 것

- **심볼이 읽은 config 필드**: 전부 이 모델의 config 클래스(또는 상속/프로퍼티/getattr 기본값)에 존재한다
- **정사각 축**: 소스에서 정사각 생성/reshape 과 대응이 확인된 축 없음
- **모듈이 읽는 config 속성**: `__init__` 에서 config 를 읽는 클래스 9개를 소스에서 확인했다. 그 목록이 각 모듈의 폭이 가질 수 있는 이름의 전부다.
- **가중치 축 ↔ 모듈 소속**: 가중치 축의 이름이 전부 그 모듈(또는 그 부모)이 실제로 읽는 config 필드에서 나왔다. 이 축들은 값이 아니라 소스로 확인된 것이다.

## 행 단위 전건 — 여기부터 읽는다

접힌 표(`<phase>.jsonl`)의 **고유 행 전부**다. 레이어 인덱스만 접었고 그 밖에는 아무것도 합치지 않았다. 아래 A/B/C 절은 (모듈, 라벨)로 접은 뷰라 **한 행 안의 어긋남이 보이지 않는다** — 실제로 외부 검토가 찾아낸 결함 세 건이 전부 그 자리에 있었다.

**한 행씩 읽고 이것만 물어라: 입력·가중치·출력이 서로 말이 되는가.**

- 같은 텐서가 `input_shape` 와 `weight_shape` 에서 다른 이름을 쓰지 않는가 (가중치는 `[out, in]` 으로 저장되고 피연산자는 전치돼 있다)
- 전치·view 처럼 **이름을 바꿀 수 없는 op** 이 이름을 바꾸지 않았는가
- 행렬곱의 수축 축이 양쪽에서 같은 이름인가 — `[m,k] @ [k,n] -> [m,n]`
- 이 모듈이 그 이름을 가질 수 있는가 (소스에서 그 `nn.Linear` 를 만드는 줄을 찾아라)

고유 행 62개.

| phase | 모듈 | op | input_shape | weight_shape | output_shape |
|---|---|---|---|---|---|
| prefill | `model.embeddings` | embedding | `[['V', 'd_model'], ['B', 'T']]` | `['V', 'd_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.norm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.mixer.in_proj` | matmul | `[['T', 'd_model'], ['d_model', '2*d_inner+2*n_g*d_state+n_h_ssm']]` | `['2*d_inner+2*n_g*d_state+n_h_ssm', 'd_model']` | `[['T', '2*d_inner+2*n_g*d_state+n_h_ssm']]` |
| prefill | `model.layers.*.mixer.conv1d` | conv1d | `[['B', 'd_inner+2*n_g*d_state', 'T'], ['d_inner+2*n_g*d_state', '1', 'd_conv'], ['d_inner+2*n_g*d_state']]` | `['d_inner+2*n_g*d_state', '1', 'd_conv']` | `[['B', 'd_inner+2*n_g*d_state', 'T+d_conv-1']]` |
| prefill | `model.layers.*.mixer.act` | silu | `[['B', 'T', 'd_inner+2*n_g*d_state']]` | `None` | `[['B', 'T', 'd_inner+2*n_g*d_state']]` |
| prefill | `model.layers.*.mixer` | exp | `[['n_h_ssm']]` | `None` | `[['n_h_ssm']]` |
| prefill | `model.layers.*.mixer` | exp | `[['B', 'n_h_ssm', '1', 'd_state', 'd_chunk']]` | `None` | `[['B', 'n_h_ssm', '1', 'd_state', 'd_chunk']]` |
| prefill | `model.layers.*.mixer` | exp | `[['B', 'n_h_ssm', '1', 'd_state']]` | `None` | `[['B', 'n_h_ssm', '1', 'd_state']]` |
| prefill | `model.layers.*.mixer` | exp | `[['B', 'n_h_ssm', '2', '2']]` | `None` | `[['B', 'n_h_ssm', '2', '2']]` |
| prefill | `model.layers.*.mixer.norm` | rmsnorm | `[['B', 'T', 'd_inner']]` | `['d_inner']` | `[['B', 'T', 'd_inner']]` |
| prefill | `model.layers.*.mixer.out_proj` | matmul | `[['T', 'd_inner'], ['d_inner', 'd_model']]` | `['d_model', 'd_inner']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_add | `[['B', 'T', 'd_model'], ['B', 'T', 'd_model']]` | `None` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.mixer.gate` | matmul | `[['T', 'd_model'], ['d_model', 'E']]` | `['E', 'd_model']` | `[['T', 'E']]` |
| prefill | `model.layers.*.mixer.gate` | sigmoid | `[['T', 'E']]` | `None` | `[['T', 'E']]` |
| prefill | `model.layers.*.mixer.fc1_latent_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_inner/n_g']]` | `['d_inner/n_g', 'd_model']` | `[['T', 'd_inner/n_g']]` |
| prefill | `model.layers.*.mixer.experts` | grouped_matmul | `[['k*T', 'd_inner/n_g'], ['E', 'd_inner/n_g', 'd_moe'], ['E']]` | `['E', 'd_moe', 'd_inner/n_g']` | `[['k*T', 'd_moe']]` |
| prefill | `model.layers.*.mixer.experts.act_fn` | relu | `[['k*T', 'd_moe']]` | `None` | `[['k*T', 'd_moe']]` |
| prefill | `model.layers.*.mixer.experts` | grouped_matmul | `[['k*T', 'd_moe'], ['E', 'd_moe', 'd_inner/n_g'], ['E']]` | `['E', 'd_inner/n_g', 'd_moe']` | `[['k*T', 'd_inner/n_g']]` |
| prefill | `model.layers.*.mixer.fc2_latent_proj` | matmul | `[['T', 'd_inner/n_g'], ['d_inner/n_g', 'd_model']]` | `['d_model', 'd_inner/n_g']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*.mixer.shared_experts.up_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_shared']]` | `['d_shared', 'd_model']` | `[['T', 'd_shared']]` |
| prefill | `model.layers.*.mixer.shared_experts.act_fn` | relu | `[['B', 'T', 'd_shared']]` | `None` | `[['B', 'T', 'd_shared']]` |
| prefill | `model.layers.*.mixer.shared_experts.down_proj` | matmul | `[['T', 'd_shared'], ['d_shared', 'd_model']]` | `['d_model', 'd_shared']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*.mixer` | elementwise_add | `[['B', 'T', 'd_model'], ['B', 'T', 'd_model']]` | `None` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.mixer.q_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_model']]` | `['d_model', 'd_model']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*.mixer.k_proj` | matmul | `[['T', 'd_model'], ['d_model', 'n_h_ssm']]` | `['n_h_ssm', 'd_model']` | `[['T', 'n_h_ssm']]` |
| prefill | `model.layers.*.mixer.v_proj` | matmul | `[['T', 'd_model'], ['d_model', 'n_h_ssm']]` | `['n_h_ssm', 'd_model']` | `[['T', 'n_h_ssm']]` |
| prefill | `model.layers.*.mixer` | batched_matmul | `[['n_h', 'T', 'd_head'], ['n_h', 'd_head', 'T']]` | `None` | `[['n_h', 'T', 'T']]` |
| prefill | `model.layers.*.mixer` | softmax | `[['B', 'n_h', 'T', 'T']]` | `None` | `[['B', 'n_h', 'T', 'T']]` |
| prefill | `model.layers.*.mixer` | batched_matmul | `[['n_h', 'T', 'T'], ['n_h', 'T', 'd_head']]` | `None` | `[['n_h', 'T', 'd_head']]` |
| prefill | `model.layers.*.mixer.o_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_model']]` | `['d_model', 'd_model']` | `[['T', 'd_model']]` |
| prefill | `model.norm_f` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `lm_head` | matmul | `[['T', 'd_model'], ['d_model', 'V']]` | `['V', 'd_model']` | `[['T', 'V']]` |
| decode | `model.embeddings` | embedding | `[['V', 'd_model'], ['B', '1']]` | `['V', 'd_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.norm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.mixer.in_proj` | matmul | `[['B', 'd_model'], ['d_model', '2*d_inner+2*n_g*d_state+n_h_ssm']]` | `['2*d_inner+2*n_g*d_state+n_h_ssm', 'd_model']` | `[['B', '2*d_inner+2*n_g*d_state+n_h_ssm']]` |
| decode | `model.layers.*.mixer.act` | silu | `[['B', 'd_inner+2*n_g*d_state']]` | `None` | `[['B', 'd_inner+2*n_g*d_state']]` |
| decode | `model.layers.*.mixer` | exp | `[['n_h_ssm']]` | `None` | `[['n_h_ssm']]` |
| decode | `model.layers.*.mixer` | exp | `[['B', 'n_h_ssm', 'd_head_ssm', 'd_state']]` | `None` | `[['B', 'n_h_ssm', 'd_head_ssm', 'd_state']]` |
| decode | `model.layers.*.mixer` | batched_matmul | `[['n_h_ssm', 'd_head_ssm', 'd_state'], ['n_h_ssm', 'd_state', 'B']]` | `None` | `[['n_h_ssm', 'd_head_ssm', 'B']]` |
| decode | `model.layers.*.mixer.norm` | rmsnorm | `[['B', '1', 'd_inner']]` | `['d_inner']` | `[['B', '1', 'd_inner']]` |
| decode | `model.layers.*.mixer.out_proj` | matmul | `[['B', 'd_inner'], ['d_inner', 'd_model']]` | `['d_model', 'd_inner']` | `[['B', 'd_model']]` |
| decode | `model.layers.*` | elementwise_add | `[['B', '1', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.mixer.gate` | matmul | `[['B', 'd_model'], ['d_model', 'E']]` | `['E', 'd_model']` | `[['B', 'E']]` |
| decode | `model.layers.*.mixer.gate` | sigmoid | `[['B', 'E']]` | `None` | `[['B', 'E']]` |
| decode | `model.layers.*.mixer.fc1_latent_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_inner/n_g']]` | `['d_inner/n_g', 'd_model']` | `[['B', 'd_inner/n_g']]` |
| decode | `model.layers.*.mixer.experts` | grouped_matmul | `[['k', 'd_inner/n_g'], ['E', 'd_inner/n_g', 'd_moe'], ['E']]` | `['E', 'd_moe', 'd_inner/n_g']` | `[['k', 'd_moe']]` |
| decode | `model.layers.*.mixer.experts.act_fn` | relu | `[['k', 'd_moe']]` | `None` | `[['k', 'd_moe']]` |
| decode | `model.layers.*.mixer.experts` | grouped_matmul | `[['k', 'd_moe'], ['E', 'd_moe', 'd_inner/n_g'], ['E']]` | `['E', 'd_inner/n_g', 'd_moe']` | `[['k', 'd_inner/n_g']]` |
| decode | `model.layers.*.mixer.fc2_latent_proj` | matmul | `[['B', 'd_inner/n_g'], ['d_inner/n_g', 'd_model']]` | `['d_model', 'd_inner/n_g']` | `[['B', 'd_model']]` |
| decode | `model.layers.*.mixer.shared_experts.up_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_shared']]` | `['d_shared', 'd_model']` | `[['B', 'd_shared']]` |
| decode | `model.layers.*.mixer.shared_experts.act_fn` | relu | `[['B', '1', 'd_shared']]` | `None` | `[['B', '1', 'd_shared']]` |
| decode | `model.layers.*.mixer.shared_experts.down_proj` | matmul | `[['B', 'd_shared'], ['d_shared', 'd_model']]` | `['d_model', 'd_shared']` | `[['B', 'd_model']]` |
| decode | `model.layers.*.mixer` | elementwise_add | `[['B', '1', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.mixer.q_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_model']]` | `['d_model', 'd_model']` | `[['B', 'd_model']]` |
| decode | `model.layers.*.mixer.k_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_h_ssm']]` | `['n_h_ssm', 'd_model']` | `[['B', 'n_h_ssm']]` |
| decode | `model.layers.*.mixer.v_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_h_ssm']]` | `['n_h_ssm', 'd_model']` | `[['B', 'n_h_ssm']]` |
| decode | `model.layers.*.mixer` | batched_matmul | `[['n_h', 'B', 'd_head'], ['n_h', 'd_head', 'T+1']]` | `None` | `[['n_h', 'B', 'T+1']]` |
| decode | `model.layers.*.mixer` | softmax | `[['B', 'n_h', '1', 'T+1']]` | `None` | `[['B', 'n_h', '1', 'T+1']]` |
| decode | `model.layers.*.mixer` | batched_matmul | `[['n_h', 'B', 'T+1'], ['n_h', 'T+1', 'd_head']]` | `None` | `[['n_h', 'B', 'd_head']]` |
| decode | `model.layers.*.mixer.o_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_model']]` | `['d_model', 'd_model']` | `[['B', 'd_model']]` |
| decode | `model.norm_f` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `lm_head` | matmul | `[['B', 'd_model'], ['d_model', 'V']]` | `['V', 'd_model']` | `[['B', 'V']]` |

## 전수 점검 — 이 모델이 쓰는 이름 전부

위 절이 '풀리지 않은 것'이라면 여기는 **전부**다. 규칙이 자신 있게 붙인 이름도 틀릴 수 있고, 그런 건 미결 목록에 절대 오르지 않는다. 한 줄씩 읽고 **그 모듈에서 그 이름이 말이 되는지** 보라.

### A. 붙은 이름 전부 (27종)

| 라벨 | 값 | 나타나는 모듈 | 축 수 |
|---|---|---|---|
| `B` |  | `model.layers.*.mixer`, `model.layers.*.norm`, `model.layers.*.mixer.gate`, `model.layers.*.mixer.norm` 외 127개 | 32832 |
| `n_h_ssm` | 256 | `model.layers.*.mixer`, `model.layers.*.mixer.k_proj`, `model.layers.*.mixer.v_proj` | 14208 |
| `T` |  | `model.layers.*.mixer`, `model.layers.*.mixer.gate`, `model.layers.*.norm`, `model.layers.*.mixer.norm` 외 127개 | 12995 |
| `d_state` | 128 | `model.layers.*.mixer` | 10800 |
| `d_model` | 8192 | `model.layers.*.norm`, `model.layers.*.mixer.gate`, `model.layers.*.mixer.in_proj`, `model.layers.*.mixer.out_proj` 외 121개 | 10634 |
| `d_head_ssm` | 64 | `model.layers.*.mixer` | 6144 |
| `d_inner/n_g` |  | `model.layers.*.mixer.experts`, `model.layers.*.mixer.norm`, `model.layers.*.mixer.fc1_latent_proj`, `model.layers.*.mixer.fc2_latent_proj` | 4512 |
| `E` | 512 | `model.layers.*.mixer.gate`, `model.layers.*.mixer.experts` | 4224 |
| `k` | 22 | `model.layers.*.mixer.experts`, `model.layers.*.mixer.gate`, `model.layers.*.mixer.experts.act_fn` | 3696 |
| `d_chunk` | 128 | `model.layers.*.mixer` | 3168 |
| `n_g_ssm` | 8 | `model.layers.*.mixer`, `model.layers.*.mixer.norm` | 2976 |
| `d_inner` |  | `model.layers.*.mixer.norm`, `model.layers.*.mixer.out_proj`, `model.layers.*.mixer` | 2688 |
| `d_inner+2*n_g*d_state` |  | `model.layers.*.mixer`, `model.layers.*.mixer.conv1d`, `model.layers.*.mixer.act` | 2592 |
| `k*T` |  | `model.layers.*.mixer.experts`, `model.layers.*.mixer.experts.act_fn` | 2448 |
| `d_shared` | 10240 | `model.layers.*.mixer.shared_experts.up_proj`, `model.layers.*.mixer.shared_experts.down_proj`, `model.layers.*.mixer.shared_experts.act_fn` | 1920 |
| `d_moe` | 5120 | `model.layers.*.mixer.experts`, `model.layers.*.mixer.experts.act_fn` | 1536 |
| `d_head` | 128 | `model.layers.*.mixer` | 1392 |
| `d_conv` | 4 | `model.layers.*.mixer`, `model.layers.*.mixer.conv1d` | 1152 |
| `n_h` | 64 | `model.layers.*.mixer` | 1128 |
| `n_h_ssm/n_g_ssm` |  | `model.layers.*.mixer` | 960 |
| `2*d_inner+2*n_g*d_state+n_h_ssm` |  | `model.layers.*.mixer.in_proj`, `model.layers.*.mixer` | 864 |
| `T+1` |  | `model.layers.*.mixer`, `model` | 603 |
| `n_g*d_state` |  | `model.layers.*.mixer` | 384 |
| `d_conv+1` |  | `model.layers.*.mixer` | 144 |
| `T+d_conv-1` |  | `model.layers.*.mixer.conv1d`, `model.layers.*.mixer` | 96 |
| `n_kv` | 2 | `model.layers.*.mixer` | 96 |
| `V` | 131072 | `lm_head`, `model.embeddings`, `(root)` | 24 |

### B. 이름 없이 남은 정수 전부 (2쌍)

**여기가 필터가 못 보던 자리다.** 정수가 남는 것 자체는 정상이다(루프 인덱스, 피연산자 개수, 브로드캐스트 축). 문제는 **이름이 있어야 하는데 없는 경우**이고, 마지막 열이 그 신호다 — 이 모델의 심볼과 값이 같다면 스코프가 그 모듈을 못 덮고 있을 수 있다. 실제로 `n_hc`(=4)가 그렇게 정수로 남아 있었다.

| 모듈 | 정수 | 축 수 | 같은 값의 심볼 |
|---|---|---|---|
| `model.layers.*.mixer` | 2 | 3912 | `n_kv` |
| `model.layers.*.mixer.gate` | 2 | 288 | `n_kv` |

### C. 모듈이 내는 출력 shape 전부 (132개 모듈 / 467종)

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
  - `[[B, 1, 1, d_state, n_h_ssm, d_chunk]]`
  - `[[B, 1, 1, d_state, n_h_ssm, d_head_ssm]]`
  - `[[B, 1, 1, n_h_ssm, d_head_ssm, d_state]]`
  - `[[B, 1, 2, d_head]]`
  - `[[B, 1, d_inner+2*n_g*d_state]]`
  - `[[B, 1, d_inner], [B, 1, n_g*d_state], [B, 1, n_g*d_state]]`
  - `[[B, 1, d_inner]]`
  - `[[B, 1, d_model]]`
  - `[[B, 1, d_state, 1, n_h_ssm, d_chunk]]`
  - `[[B, 1, d_state, d_chunk, n_h_ssm, 1]]`
  - `[[B, 1, d_state, d_chunk, n_h_ssm, d_head]]`
  - `[[B, 1, d_state, d_chunk, n_h_ssm, d_head_ssm]]`
  - `[[B, 1, d_state, d_chunk, n_h_ssm]]`
  - `[[B, 1, d_state, n_h_ssm, 1, d_chunk]]`
  - `[[B, 1, d_state, n_h_ssm, 1]]`
  - `[[B, 1, d_state, n_h_ssm, d_chunk]]`
  - `[[B, 1, d_state, n_h_ssm, d_head_ssm, d_chunk]]`
  - `[[B, 1, d_state, n_h_ssm, d_head_ssm]]`
  - `[[B, 1, d_state, n_h_ssm]]`
  - `[[B, 1, n_h, d_head]]`
  - `[[B, 1, n_h_ssm, d_head_ssm, d_state]]`
  - `[[B, 1, n_h_ssm, d_state, 1, d_head_ssm]]`
  - `[[B, 1, n_h_ssm, d_state, d_chunk, 1]]`
  - `[[B, 1, n_h_ssm, d_state, d_chunk, d_head_ssm]]`
  - `[[B, 1, n_h_ssm, d_state, d_chunk]]`
  - `[[B, 1, n_h_ssm, d_state, d_head_ssm]]`
  - `[[B, 1, n_h_ssm]]`
  - `[[B, 2, 1, T+1, d_head]]`
  - `[[B, 2, 1, T, d_head]]`
  - `[[B, 2, 1, d_head]]`
  - `[[B, 2, T+1, d_head]]`
  - `[[B, 2, T, d_head]]`
  - `[[B, 2, n_h_ssm, d_head_ssm, d_state]]`
  - `[[B, 2, n_h_ssm/n_g_ssm, T+1, d_head]]`
  - `[[B, 2, n_h_ssm/n_g_ssm, T, d_head]]`
  - `[[B, T, 0], [B, T, 0], [B, T, d_inner], [B, T, d_inner+2*n_g*d_state], [B, T, n_h_ssm]]`
  - `[[B, T, d_inner+2*n_g*d_state]]`
  - `[[B, T, d_inner], [B, T, n_g*d_state], [B, T, n_g*d_state]]`
  - `[[B, T, d_inner]]`
  - `[[B, T, d_model]]`
  - `[[B, T, n_g_ssm, 1, d_state]]`
  - `[[B, T, n_g_ssm, d_state]]`
  - `[[B, T, n_g_ssm, n_h_ssm/n_g_ssm, d_state]]`
  - `[[B, T, n_h, d_head]]`
  - `[[B, T, n_h_ssm, 1]]`
  - `[[B, T, n_h_ssm, d_head_ssm]]`
  - `[[B, T, n_h_ssm, d_state]]`
  - `[[B, T, n_h_ssm]]`
  - `[[B, T, n_kv, d_head]]`
  - `[[B, d_inner+2*n_g*d_state, 1]]`
  - `[[B, d_inner+2*n_g*d_state, T]]`
  - `[[B, d_inner+2*n_g*d_state, d_conv+1]]`
  - `[[B, d_inner+2*n_g*d_state, d_conv]]`
  - `[[B, d_inner+2*n_g*d_state]]`
  - `[[B, d_inner]]`
  - `[[B, d_model]]`
  - `[[B, d_state, n_h_ssm, d_chunk]]`
  - `[[B, d_state, n_h_ssm, d_head_ssm]]`
  - `[[B, d_state, n_h_ssm]]`
  - `[[B, n_g_ssm, 1, d_state]]`
  - `[[B, n_g_ssm, d_state]]`
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
  - `[[B, n_h_ssm, 1, d_state, 1]]`
  - `[[B, n_h_ssm, 1, d_state, d_chunk]]`
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
  - `[[B, n_kv, T, d_head]]`
  - `[[T, d_model]]`
  - `[[d_inner+2*n_g*d_state, d_conv]]`
  - `[[d_state, d_chunk]]`
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
  - `[[T, 1, 2], [T, 1, 2]]`
  - `[[T, 1, E]]`
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
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[T, d_model]]`
  - `[[d_model, d_model]]`
- `model.layers.*.mixer.out_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_inner]]`
  - `[[B, d_model]]`
  - `[[T, d_inner]]`
  - `[[T, d_model]]`
  - `[[d_inner, d_model]]`
- `model.layers.*.mixer.q_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[T, d_model]]`
  - `[[d_model, d_model]]`
- `model.layers.*.mixer.shared_experts.act_fn`
  - `[[B, 1, d_shared]]`
  - `[[B, T, d_shared]]`
- `model.layers.*.mixer.shared_experts.down_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[B, d_shared]]`
  - `[[T, d_model]]`
  - `[[T, d_shared]]`
  - `[[d_shared, d_model]]`
- `model.layers.*.mixer.shared_experts.up_proj`
  - `[[B, 1, d_shared]]`
  - `[[B, T, d_shared]]`
  - `[[B, d_model]]`
  - `[[B, d_shared]]`
  - `[[T, d_model]]`
  - `[[T, d_shared]]`
  - `[[d_model, d_shared]]`
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
