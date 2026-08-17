# 검토 의뢰서 — Qwen/Qwen3-Next-80B-A3B-Instruct

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `qwen3_next`
- 판단 필요: **11건**

## 증거 — 이미 받아둔 실제 소스

- `develop/sources/modeling_qwen3_next.py` — 있음, 이 파일을 열어서 판정한다
- `develop/sources/configuration_qwen3_next.py` — 있음, 이 파일을 열어서 판정한다

- 온라인 원본: https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_next

그 밖의 재료: `full/review.md`(리뷰 패킷 — shape 별 실제 행 표본), `structure.yaml`(이 모델의 심볼 표), `full/<phase>.csv`(전체 operator 표).

## 판단이 필요한 것

### 2. 이 정사각 축이 정말 같은 이름 두 번인가

`[..., X, X]` 로 렌더됐는데, 그 이름이 읽은 config 필드에서 나온 정사각 reshape 을 modeling 소스에서 찾지 못했다. 두 축 크기가 우연히 같은 것일 수 있다.

- `d_head_lin_k`
- `d_head_lin_v`

### 6. 값이 겹쳐 **임의로** 고른 축

두 심볼이 같은 값을 갖는 자리다. 규칙에는 고를 근거가 없고, 이긴 쪽은 전역 우선순위 — 즉 **관례**로 정해졌다. 이름이 맞을 수도 있지만 파이프라인은 그걸 알지 못한다. 표에서는 확신 있는 라벨과 똑같이 보인다.

**소스를 열어 어느 쪽인지 확정하는 것이 여기서 할 일이다.** 확정되면 `rules/label_overrides.yaml` 에 근거와 함께 못 박는다(review/05-overrides.md). 출신으로만 구별되는 경우라면 그렇게 적고 `open` 으로 남긴다.

- `d_head_lin_k vs d_head_lin_v` in `model.layers.*.linear_attn` — 값 128 를 두고 후보가 2개, 12204축
- `E vs d_moe` in `model.layers.*.mlp.experts` — 값 512 를 두고 후보가 2개, 2020축
- `d_head_lin_k vs d_head_lin_v` in `model.layers.*.linear_attn.norm` — 값 128 를 두고 후보가 2개, 1514축
- `E vs d_moe` in `model.layers.*.mlp.shared_expert.gate_proj` — 값 512 를 두고 후보가 2개, 578축
- `E vs d_moe` in `model.layers.*.mlp.shared_expert.up_proj` — 값 512 를 두고 후보가 2개, 578축
- `E vs d_moe` in `model.layers.*.mlp.shared_expert.down_proj` — 값 512 를 두고 후보가 2개, 578축
- `E vs d_moe` in `model.layers.*.mlp.shared_expert` — 값 512 를 두고 후보가 2개, 288축
- `E vs d_moe` in `model.layers.*.mlp.shared_expert.act_fn` — 값 512 를 두고 후보가 2개, 192축
- `E vs d_moe` in `model.layers.*.mlp.experts.act_fn` — 값 512 를 두고 후보가 2개, 192축

### 0. 규칙이 끝내지 못한 축 — **여기부터 답한다**

값으로는 결정할 수 없어 파이프라인이 판단을 넘긴 자리다. 세 가지뿐이다:
`tie`(두 심볼이 같은 값이라 관례로 골랐다) · `heur`(등록 규칙이 없어 산술로 지어냈다) · `bare`(이름을 못 붙였는데 크기가 커서 진짜 차원일 수 있다).

**답이 나오면 `override_stub` 을 채워 `rules/label_overrides.yaml` 에 넣는다.** `spread: class` 라 그 축이 지나는 모든 자리가 한 번에 바뀐다 — 모듈 경계에서 멈추지 않는다(그것이 예전에 교정을 막던 유일한 이유였다).

**값이 같은 심볼이 여럿이면 값으로는 영원히 못 가른다. shape 안의 위치가 말해 준다** — `[B, n_h, T, d_head]` 의 축 1 은 head 개수, 축 3 은 head 폭이다.

아래 `shape` 과 `축` 은 **그 축을 처음 만든 자리(앵커)** 의 것이다. 초안은 `shape`/`axis`/`field`/`shape_index`/`op_type`/`nth` 여섯으로 그 앵커를 지목한다 — `shape`+`axis` 만으로는 부족하다(Kimi 의 `[B, n_h, T, d_nope]` 축 3 은 **366개 등가류**에 걸쳐 있다: q 의 q_pass, KV 의 k_nope, value_states …). `nth` 는 그 모듈 안에서 같은 op_type 의 몇 번째인지다 — MLA 는 `self_attn` 안에 `split_with_sizes` 가 q용·kv용 둘이라 그것 없이는 못 가른다.

**유일성은 실제로 돌려 봐서 검증한다**: 그 조건에 맞는 자리들이 몇 개의 등가류에 속하는지 세고, **한 레이어 안에서 둘 이상**이면 `stub_ambiguous` 를 붙인다. 그 초안은 쓰지 말고 `open` 으로 남길 것.

| 왜 | 모듈 | 크기 | 지금 이름 | 후보 | 축 | 앵커 shape | 축 수 |
|---|---|---|---|---|---|---|---|
| `tie` | `model.layers.*.mlp.experts.act_fn` | 512 | `d_moe` | `E`, `d_moe` | 1 | `[k*T, d_moe]` | 288 |
| `tie` | `model.layers.*.mlp.experts.act_fn` | 512 | `d_moe` | `E`, `d_moe` | 1 | `[k, d_moe]` | 288 |
| `tie` | `model.layers.*.linear_attn` | 128 | `d_head_lin_k` | `d_head_lin_k`, `d_head_lin_v` | 2 | `[B, n_h_lin_v, d_head_lin_k]` | 252 |
| `tie` | `model.layers.*.mlp.experts` | 512 | `d_moe` | `E`, `d_moe` | 0 | `[d_moe]` | 240 |
| `tie` | `model.layers.*.linear_attn` | 128 | `d_head_lin_k` | `d_head_lin_k`, `d_head_lin_v` | 3 | `[B, n_h_lin_v, T, d_head_lin_k]` | 216 |
| `tie` | `model.layers.*.linear_attn` | 128 | `d_head_lin_k` | `d_head_lin_k`, `d_head_lin_v` | 3 | `[B, n_h_lin_v, 1, d_head_lin_k]` | 216 |
| `tie` | `model.layers.*.linear_attn` | 128 | `d_head_lin_k` | `d_head_lin_k`, `d_head_lin_v` | 3 | `[B, n_h_lin_v, d_chunk, d_head_lin_k]` | 180 |
| `tie` | `model.layers.*.linear_attn` | 128 | `d_head_lin_k` | `d_head_lin_k`, `d_head_lin_v` | 3 | `[B, T, n_h_lin_k, d_head_lin_k]` | 144 |
| `tie` | `model.layers.*.linear_attn` | 128 | `d_head_lin_k` | `d_head_lin_k`, `d_head_lin_v` | 3 | `[B, 1, n_h_lin_k, d_head_lin_k]` | 144 |
| `tie` | `model.layers.*.mlp.experts` | 512 | `d_moe` | `E`, `d_moe` | 0 | `[d_moe, d_model, 2*d_moe]` | 96 |
| `tie` | `model.layers.*.mlp.experts` | 512 | `d_moe` | `E`, `d_moe` | 0 | `[d_moe, E, d_model]` | 96 |
| `tie` | `model.layers.*.mlp.experts` | 512 | `E` | `E`, `d_moe` | 1 | `[d_moe, E, d_model]` | 96 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 1, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 2, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 3, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 4, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 5, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 6, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 7, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 8, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 9, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 10, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 11, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 12, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 13, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 14, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 15, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 16, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 17, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 18, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 19, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 20, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 21, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 22, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 23, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 24, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 25, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 26, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 27, 64]` | 72 |
| `bare` | `model.layers.*.linear_attn` | 64 | `64` | — | 4 | `[B, n_h_lin_v, 1, 28, 64]` | 72 |

**고칠 것과 맞는 것 둘 다 적는다.** 이름이 틀렸으면 아래 초안의 `to`/`source` 를 채워 `rules/label_overrides.yaml` 에, **지금 이름이 맞으면** 같은 앵커에 `to` 대신 `label: <지금 이름>` 과 `source` 를 적어 `rules/label_confirmed.yaml` 에 넣는다. 확인을 적지 않으면 그 축은 재생성마다 다시 질문으로 올라온다.

초안(그대로 복사해 `to` 와 `source` 만 채운다):

```yaml
  - model: Qwen__Qwen3-Next-80B-A3B-Instruct
    module: 'experts\.act_fn$'
    spread: class
    shape: ["k*T", "d_moe"]
    axis: 1
    field: i
    shape_index: 0
    op_type: silu
    nth: 0
    from: d_moe
    to: <소스가 말하는 이름>
    expect: 512
    source: <modeling_*.py:줄 인용>
  - model: Qwen__Qwen3-Next-80B-A3B-Instruct
    module: 'experts\.act_fn$'
    spread: class
    shape: ["k", "d_moe"]
    axis: 1
    field: i
    shape_index: 0
    op_type: silu
    nth: 0
    from: d_moe
    to: <소스가 말하는 이름>
    expect: 512
    source: <modeling_*.py:줄 인용>
  - model: Qwen__Qwen3-Next-80B-A3B-Instruct
    module: 'linear_attn$'
    spread: class
    shape: ["B", "n_h_lin_v", "d_head_lin_k"]
    axis: 2
    field: o
    shape_index: 0
    op_type: select
    nth: 1
    from: d_head_lin_k
    to: <소스가 말하는 이름>
    expect: 128
    source: <modeling_*.py:줄 인용>
  - model: Qwen__Qwen3-Next-80B-A3B-Instruct
    module: 'experts$'
    spread: class
    shape: ["d_moe"]
    axis: 0
    field: o
    shape_index: 0
    op_type: histc
    nth: 0
    from: d_moe
    to: <소스가 말하는 이름>
    expect: 512
    source: <modeling_*.py:줄 인용>
  - model: Qwen__Qwen3-Next-80B-A3B-Instruct
    module: 'linear_attn$'
    spread: class
    shape: ["B", "n_h_lin_v", "T", "d_head_lin_k"]
    axis: 3
    field: o
    shape_index: 0
    op_type: transpose
    nth: 2
    from: d_head_lin_k
    to: <소스가 말하는 이름>
    expect: 128
    source: <modeling_*.py:줄 인용>
  - model: Qwen__Qwen3-Next-80B-A3B-Instruct
    module: 'linear_attn$'
    spread: class
    shape: ["B", "n_h_lin_v", "1", "d_head_lin_k"]
    axis: 3
    field: o
    shape_index: 0
    op_type: transpose
    nth: 2
    from: d_head_lin_k
    to: <소스가 말하는 이름>
    expect: 128
    source: <modeling_*.py:줄 인용>
```

## 기계적으로 이미 확인된 것 — 다시 묻지 말 것

- **심볼이 읽은 config 필드**: 전부 이 모델의 config 클래스(또는 상속/프로퍼티/getattr 기본값)에 존재한다
- **정사각 축**: 소스에서 정사각 생성/reshape 과 대응이 확인된 축 없음
- **모듈이 읽는 config 속성**: `__init__` 에서 config 를 읽는 클래스 10개를 소스에서 확인했다. 그 목록이 각 모듈의 폭이 가질 수 있는 이름의 전부다.
- **가중치 축 ↔ 모듈 소속**: 가중치 축의 이름이 전부 그 모듈(또는 그 부모)이 실제로 읽는 config 필드에서 나왔다. 이 축들은 값이 아니라 소스로 확인된 것이다.

## 행 단위 전건 — 여기부터 읽는다

접힌 표(`<phase>.jsonl`)의 **고유 행 전부**다. 레이어 인덱스만 접었고 그 밖에는 아무것도 합치지 않았다. 아래 A/B/C 절은 (모듈, 라벨)로 접은 뷰라 **한 행 안의 어긋남이 보이지 않는다** — 실제로 외부 검토가 찾아낸 결함 세 건이 전부 그 자리에 있었다.

**한 행씩 읽고 이것만 물어라: 입력·가중치·출력이 서로 말이 되는가.**

- 같은 텐서가 `input_shape` 와 `weight_shape` 에서 다른 이름을 쓰지 않는가 (가중치는 `[out, in]` 으로 저장되고 피연산자는 전치돼 있다)
- 전치·view 처럼 **이름을 바꿀 수 없는 op** 이 이름을 바꾸지 않았는가
- 행렬곱의 수축 축이 양쪽에서 같은 이름인가 — `[m,k] @ [k,n] -> [m,n]`
- 이 모듈이 그 이름을 가질 수 있는가 (소스에서 그 `nn.Linear` 를 만드는 줄을 찾아라)

고유 행 95개.

| phase | 모듈 | op | input_shape | weight_shape | output_shape |
|---|---|---|---|---|---|
| prefill | `model.embed_tokens` | embedding | `[['V', 'd_model'], ['B', 'T']]` | `['V', 'd_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.input_layernorm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.linear_attn.in_proj_qkvz` | matmul | `[['T', 'd_model'], ['d_model', '2*n_k*d_k+2*n_v*d_v']]` | `['2*n_k*d_k+2*n_v*d_v', 'd_model']` | `[['T', '2*n_k*d_k+2*n_v*d_v']]` |
| prefill | `model.layers.*.linear_attn.in_proj_ba` | matmul | `[['T', 'd_model'], ['d_model', 'd_chunk']]` | `['d_chunk', 'd_model']` | `[['T', 'd_chunk']]` |
| prefill | `model.layers.*.linear_attn` | concat | `[['B', 'T', 'd_model'], ['B', 'T', 'd_model'], ['B', 'T', 'n_v*d_v']]` | `None` | `[['B', 'T', '2*n_h*d_head']]` |
| prefill | `model.layers.*.linear_attn.conv1d` | conv1d | `[['B', '2*n_h*d_head', 'T'], ['2*n_h*d_head', '1', 'd_conv_lin']]` | `['2*n_h*d_head', '1', 'd_conv_lin']` | `[['B', '2*n_h*d_head', 'n_h+2*n_kv']]` |
| prefill | `model.layers.*.linear_attn` | silu | `[['B', '2*n_h*d_head', 'T']]` | `None` | `[['B', '2*n_h*d_head', 'T']]` |
| prefill | `model.layers.*.linear_attn` | sigmoid | `[['B', 'T', 'n_h_lin_v']]` | `None` | `[['B', 'T', 'n_h_lin_v']]` |
| prefill | `model.layers.*.linear_attn` | exp | `[['n_h_lin_v']]` | `None` | `[['n_h_lin_v']]` |
| prefill | `model.layers.*.linear_attn` | exp | `[['B', 'n_h_lin_v', '1', 'd_chunk', 'd_chunk']]` | `None` | `[['B', 'n_h_lin_v', '1', 'd_chunk', 'd_chunk']]` |
| prefill | `model.layers.*.linear_attn` | batched_matmul | `[['n_h_lin_v', 'd_chunk', 'd_head_lin_k'], ['n_h_lin_v', 'd_head_lin_k', 'd_chunk']]` | `None` | `[['n_h_lin_v', 'd_chunk', 'd_chunk']]` |
| prefill | `model.layers.*.linear_attn` | batched_matmul | `[['n_h_lin_v', 'd_chunk', 'd_chunk'], ['n_h_lin_v', 'd_chunk', 'd_head_lin_v']]` | `None` | `[['n_h_lin_v', 'd_chunk', 'd_head_lin_v']]` |
| prefill | `model.layers.*.linear_attn` | exp | `[['B', 'n_h_lin_v', '1', 'd_chunk']]` | `None` | `[['B', 'n_h_lin_v', '1', 'd_chunk']]` |
| prefill | `model.layers.*.linear_attn` | batched_matmul | `[['n_h_lin_v', 'd_chunk', 'd_chunk'], ['n_h_lin_v', 'd_chunk', 'd_head_lin_k']]` | `None` | `[['n_h_lin_v', 'd_chunk', 'd_head_lin_k']]` |
| prefill | `model.layers.*.linear_attn` | batched_matmul | `[['n_h_lin_v', 'd_chunk', 'd_head_lin_k'], ['n_h_lin_v', 'd_head_lin_k', 'd_head_lin_v']]` | `None` | `[['n_h_lin_v', 'd_chunk', 'd_head_lin_v']]` |
| prefill | `model.layers.*.linear_attn` | exp | `[['B', 'n_h_lin_v', 'd_chunk', '1']]` | `None` | `[['B', 'n_h_lin_v', 'd_chunk', '1']]` |
| prefill | `model.layers.*.linear_attn` | exp | `[['B', 'n_h_lin_v', '1', '1']]` | `None` | `[['B', 'n_h_lin_v', '1', '1']]` |
| prefill | `model.layers.*.linear_attn` | exp | `[['B', 'n_h_lin_v', 'd_chunk']]` | `None` | `[['B', 'n_h_lin_v', 'd_chunk']]` |
| prefill | `model.layers.*.linear_attn` | batched_matmul | `[['n_h_lin_v', 'd_head_lin_k', 'd_chunk'], ['n_h_lin_v', 'd_chunk', 'd_head_lin_v']]` | `None` | `[['n_h_lin_v', 'd_head_lin_k', 'd_head_lin_v']]` |
| prefill | `model.layers.*.linear_attn.norm` | rmsnorm | `[['n_h_lin_v*T', 'd_head_lin_v']]` | `['d_head_lin_v']` | `[['n_h_lin_v*T', 'd_head_lin_v']]` |
| prefill | `model.layers.*.linear_attn.out_proj` | matmul | `[['T', 'n_v*d_v'], ['n_v*d_v', 'd_model']]` | `['d_model', 'n_v*d_v']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_add | `[['B', 'T', 'd_model'], ['B', 'T', 'd_model']]` | `None` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.post_attention_layernorm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.mlp.shared_expert.gate_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_moe']]` | `['d_moe', 'd_model']` | `[['T', 'd_moe']]` |
| prefill | `model.layers.*.mlp.shared_expert.act_fn` | silu | `[['T', 'd_moe']]` | `None` | `[['T', 'd_moe']]` |
| prefill | `model.layers.*.mlp.shared_expert.up_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_moe']]` | `['d_moe', 'd_model']` | `[['T', 'd_moe']]` |
| prefill | `model.layers.*.mlp.shared_expert` | elementwise_mul | `[['T', 'd_moe'], ['T', 'd_moe']]` | `None` | `[['T', 'd_moe']]` |
| prefill | `model.layers.*.mlp.shared_expert.down_proj` | matmul | `[['T', 'd_moe'], ['d_moe', 'd_model']]` | `['d_model', 'd_moe']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*.mlp.gate` | matmul | `[['T', 'd_model'], ['d_model', 'E']]` | `['E', 'd_model']` | `[['T', 'E']]` |
| prefill | `model.layers.*.mlp.gate` | softmax | `[['T', 'E']]` | `None` | `[['T', 'E']]` |
| prefill | `model.layers.*.mlp.experts` | grouped_matmul | `[['k*T', 'd_model'], ['d_moe', 'd_model', '2*d_moe'], ['d_moe']]` | `['d_moe', '2*d_moe', 'd_model']` | `[['k*T', '2*d_moe']]` |
| prefill | `model.layers.*.mlp.experts.act_fn` | silu | `[['k*T', 'd_moe']]` | `None` | `[['k*T', 'd_moe']]` |
| prefill | `model.layers.*.mlp.experts` | elementwise_mul | `[['k*T', 'd_moe'], ['k*T', 'd_moe']]` | `None` | `[['k*T', 'd_moe']]` |
| prefill | `model.layers.*.mlp.experts` | grouped_matmul | `[['k*T', 'd_moe'], ['d_moe', 'E', 'd_model'], ['d_moe']]` | `['d_moe', 'd_model', 'E']` | `[['k*T', 'd_model']]` |
| prefill | `model.layers.*.mlp.experts` | elementwise_mul | `[['k*T', 'd_model'], ['k*T', 'B']]` | `None` | `[['k*T', 'd_model']]` |
| prefill | `model.layers.*.mlp.experts` | sum | `[['T', 'k', 'd_model']]` | `None` | `[['T', 'd_model']]` |
| prefill | `model.layers.*.mlp.shared_expert_gate` | matmul | `[['T', 'd_model'], ['d_model', '1']]` | `['1', 'd_model']` | `[['T', '1']]` |
| prefill | `model.layers.*.mlp` | sigmoid | `[['T', '1']]` | `None` | `[['T', '1']]` |
| prefill | `model.layers.*.mlp` | elementwise_mul | `[['T', '1'], ['T', 'd_model']]` | `None` | `[['T', 'd_model']]` |
| prefill | `model.layers.*.mlp` | elementwise_add | `[['T', 'd_model'], ['T', 'd_model']]` | `None` | `[['T', 'd_model']]` |
| prefill | `model.layers.*.self_attn.q_proj` | matmul | `[['T', 'd_model'], ['d_model', '2*n_h*d_head']]` | `['2*n_h*d_head', 'd_model']` | `[['T', '2*n_h*d_head']]` |
| prefill | `model.layers.*.self_attn.q_norm` | rmsnorm | `[['B', 'T', 'n_h', 'd_head']]` | `['d_head']` | `[['B', 'T', 'n_h', 'd_head']]` |
| prefill | `model.layers.*.self_attn.k_proj` | matmul | `[['T', 'd_model'], ['d_model', 'n_kv*d_head']]` | `['n_kv*d_head', 'd_model']` | `[['T', 'n_kv*d_head']]` |
| prefill | `model.layers.*.self_attn.k_norm` | rmsnorm | `[['B', 'T', 'n_kv', 'd_head']]` | `['d_head']` | `[['B', 'T', 'n_kv', 'd_head']]` |
| prefill | `model.layers.*.self_attn.v_proj` | matmul | `[['T', 'd_model'], ['d_model', 'n_kv*d_head']]` | `['n_kv*d_head', 'd_model']` | `[['T', 'n_kv*d_head']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'T', 'd_head'], ['n_h', 'd_head', 'T']]` | `None` | `[['n_h', 'T', 'T']]` |
| prefill | `model.layers.*.self_attn` | softmax | `[['B', 'n_h', 'T', 'T']]` | `None` | `[['B', 'n_h', 'T', 'T']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'T', 'T'], ['n_h', 'T', 'd_head']]` | `None` | `[['n_h', 'T', 'd_head']]` |
| prefill | `model.layers.*.self_attn` | sigmoid | `[['B', 'T', 'n_h*d_head']]` | `None` | `[['B', 'T', 'n_h*d_head']]` |
| prefill | `model.layers.*.self_attn.o_proj` | matmul | `[['T', 'n_h*d_head'], ['n_h*d_head', 'd_model']]` | `['d_model', 'n_h*d_head']` | `[['T', 'd_model']]` |
| prefill | `model.norm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `lm_head` | matmul | `[['T', 'd_model'], ['d_model', 'V']]` | `['V', 'd_model']` | `[['T', 'V']]` |
| decode | `model.embed_tokens` | embedding | `[['V', 'd_model'], ['B', '1']]` | `['V', 'd_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.input_layernorm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.linear_attn.in_proj_qkvz` | matmul | `[['B', 'd_model'], ['d_model', '2*n_k*d_k+2*n_v*d_v']]` | `['2*n_k*d_k+2*n_v*d_v', 'd_model']` | `[['B', '2*n_k*d_k+2*n_v*d_v']]` |
| decode | `model.layers.*.linear_attn.in_proj_ba` | matmul | `[['B', 'd_model'], ['d_model', 'd_chunk']]` | `['d_chunk', 'd_model']` | `[['B', 'd_chunk']]` |
| decode | `model.layers.*.linear_attn` | concat | `[['B', '1', 'd_model'], ['B', '1', 'd_model'], ['B', '1', 'n_v*d_v']]` | `None` | `[['B', '1', '2*n_h*d_head']]` |
| decode | `model.layers.*.linear_attn` | conv1d | `[['B', '2*n_h*d_head', '5'], ['2*n_h*d_head', 'B', 'd_conv_lin']]` | `None` | `[['B', '2*n_h*d_head', 'n_v/n_k']]` |
| decode | `model.layers.*.linear_attn` | silu | `[['B', '2*n_h*d_head', '1']]` | `None` | `[['B', '2*n_h*d_head', '1']]` |
| decode | `model.layers.*.linear_attn` | sigmoid | `[['B', '1', 'n_h_lin_v']]` | `None` | `[['B', '1', 'n_h_lin_v']]` |
| decode | `model.layers.*.linear_attn` | exp | `[['n_h_lin_v']]` | `None` | `[['n_h_lin_v']]` |
| decode | `model.layers.*.linear_attn` | exp | `[['B', 'n_h_lin_v']]` | `None` | `[['B', 'n_h_lin_v']]` |
| decode | `model.layers.*.linear_attn.norm` | rmsnorm | `[['n_h_lin_v', 'd_head_lin_v']]` | `['d_head_lin_v']` | `[['n_h_lin_v', 'd_head_lin_v']]` |
| decode | `model.layers.*.linear_attn.out_proj` | matmul | `[['B', 'n_v*d_v'], ['n_v*d_v', 'd_model']]` | `['d_model', 'n_v*d_v']` | `[['B', 'd_model']]` |
| decode | `model.layers.*` | elementwise_add | `[['B', '1', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.post_attention_layernorm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.mlp.shared_expert.gate_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_moe']]` | `['d_moe', 'd_model']` | `[['B', 'd_moe']]` |
| decode | `model.layers.*.mlp.shared_expert.act_fn` | silu | `[['B', 'd_moe']]` | `None` | `[['B', 'd_moe']]` |
| decode | `model.layers.*.mlp.shared_expert.up_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_moe']]` | `['d_moe', 'd_model']` | `[['B', 'd_moe']]` |
| decode | `model.layers.*.mlp.shared_expert` | elementwise_mul | `[['B', 'd_moe'], ['B', 'd_moe']]` | `None` | `[['B', 'd_moe']]` |
| decode | `model.layers.*.mlp.shared_expert.down_proj` | matmul | `[['B', 'd_moe'], ['d_moe', 'd_model']]` | `['d_model', 'd_moe']` | `[['B', 'd_model']]` |
| decode | `model.layers.*.mlp.gate` | matmul | `[['B', 'd_model'], ['d_model', 'E']]` | `['E', 'd_model']` | `[['B', 'E']]` |
| decode | `model.layers.*.mlp.gate` | softmax | `[['B', 'E']]` | `None` | `[['B', 'E']]` |
| decode | `model.layers.*.mlp.experts` | grouped_matmul | `[['k', 'd_model'], ['d_moe', 'd_model', '2*d_moe'], ['d_moe']]` | `['d_moe', '2*d_moe', 'd_model']` | `[['k', '2*d_moe']]` |
| decode | `model.layers.*.mlp.experts.act_fn` | silu | `[['k', 'd_moe']]` | `None` | `[['k', 'd_moe']]` |
| decode | `model.layers.*.mlp.experts` | elementwise_mul | `[['k', 'd_moe'], ['k', 'd_moe']]` | `None` | `[['k', 'd_moe']]` |
| decode | `model.layers.*.mlp.experts` | grouped_matmul | `[['k', 'd_moe'], ['d_moe', 'E', 'd_model'], ['d_moe']]` | `['d_moe', 'd_model', 'E']` | `[['k', 'd_model']]` |
| decode | `model.layers.*.mlp.experts` | elementwise_mul | `[['k', 'd_model'], ['k', 'B']]` | `None` | `[['k', 'd_model']]` |
| decode | `model.layers.*.mlp.experts` | sum | `[['B', 'k', 'd_model']]` | `None` | `[['B', 'd_model']]` |
| decode | `model.layers.*.mlp.shared_expert_gate` | matmul | `[['B', 'd_model'], ['d_model', '1']]` | `['1', 'd_model']` | `[['B', '1']]` |
| decode | `model.layers.*.mlp` | sigmoid | `[['B', '1']]` | `None` | `[['B', '1']]` |
| decode | `model.layers.*.mlp` | elementwise_mul | `[['B', '1'], ['B', 'd_model']]` | `None` | `[['B', 'd_model']]` |
| decode | `model.layers.*.mlp` | elementwise_add | `[['B', 'd_model'], ['B', 'd_model']]` | `None` | `[['B', 'd_model']]` |
| decode | `model.layers.*.self_attn.q_proj` | matmul | `[['B', 'd_model'], ['d_model', '2*n_h*d_head']]` | `['2*n_h*d_head', 'd_model']` | `[['B', '2*n_h*d_head']]` |
| decode | `model.layers.*.self_attn.q_norm` | rmsnorm | `[['B', '1', 'n_h', 'd_head']]` | `['d_head']` | `[['B', '1', 'n_h', 'd_head']]` |
| decode | `model.layers.*.self_attn.k_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_kv*d_head']]` | `['n_kv*d_head', 'd_model']` | `[['B', 'n_kv*d_head']]` |
| decode | `model.layers.*.self_attn.k_norm` | rmsnorm | `[['B', '1', 'n_kv', 'd_head']]` | `['d_head']` | `[['B', '1', 'n_kv', 'd_head']]` |
| decode | `model.layers.*.self_attn.v_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_kv*d_head']]` | `['n_kv*d_head', 'd_model']` | `[['B', 'n_kv*d_head']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'B', 'd_head'], ['n_h', 'd_head', 'T+1']]` | `None` | `[['n_h', 'B', 'T+1']]` |
| decode | `model.layers.*.self_attn` | softmax | `[['B', 'n_h', '1', 'T+1']]` | `None` | `[['B', 'n_h', '1', 'T+1']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'B', 'T+1'], ['n_h', 'T+1', 'd_head']]` | `None` | `[['n_h', 'B', 'd_head']]` |
| decode | `model.layers.*.self_attn` | sigmoid | `[['B', '1', 'n_h*d_head']]` | `None` | `[['B', '1', 'n_h*d_head']]` |
| decode | `model.layers.*.self_attn.o_proj` | matmul | `[['B', 'n_h*d_head'], ['n_h*d_head', 'd_model']]` | `['d_model', 'n_h*d_head']` | `[['B', 'd_model']]` |
| decode | `model.norm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `lm_head` | matmul | `[['B', 'd_model'], ['d_model', 'V']]` | `['V', 'd_model']` | `[['B', 'V']]` |

## 전수 점검 — 이 모델이 쓰는 이름 전부

위 절이 '풀리지 않은 것'이라면 여기는 **전부**다. 규칙이 자신 있게 붙인 이름도 틀릴 수 있고, 그런 건 미결 목록에 절대 오르지 않는다. 한 줄씩 읽고 **그 모듈에서 그 이름이 말이 되는지** 보라.

### A. 붙은 이름 전부 (34종)

| 라벨 | 값 | 나타나는 모듈 | 축 수 |
|---|---|---|---|
| `B` |  | `model.layers.*.linear_attn`, `model.layers.*.self_attn`, `model.layers.*.input_layernorm`, `model.layers.*.post_attention_layernorm` 외 73개 | 99332 |
| `n_h_lin_v` | 32 | `model.layers.*.linear_attn`, `model.layers.*.linear_attn.norm` | 84096 |
| `d_chunk` | 64 | `model.layers.*.linear_attn`, `model.layers.*.linear_attn.in_proj_ba` | 32796 |
| `T` |  | `model.layers.*.linear_attn`, `model.layers.*.self_attn`, `model.layers.*.mlp.gate`, `model.layers.*.input_layernorm` 외 72개 | 13448 |
| `d_model` | 2048 | `model.layers.*.mlp.experts`, `model.layers.*.input_layernorm`, `model.layers.*.post_attention_layernorm`, `model.layers.*.mlp` 외 65개 | 13210 |
| `d_head_lin_k` | 128 | `model.layers.*.linear_attn` | 8460 |
| `d_head_lin_v` | 128 | `model.layers.*.linear_attn`, `model.layers.*.linear_attn.norm` | 6660 |
| `d_moe` | 512 | `model.layers.*.mlp.experts`, `model.layers.*.mlp.shared_expert.gate_proj`, `model.layers.*.mlp.shared_expert.up_proj`, `model.layers.*.mlp.shared_expert.down_proj` 외 3개 | 4416 |
| `k` | 10 | `model.layers.*.mlp.experts`, `model.layers.*.mlp.gate`, `model.layers.*.mlp.experts.act_fn` | 3696 |
| `n_h_lin_k` | 16 | `model.layers.*.linear_attn` | 3168 |
| `k*T` |  | `model.layers.*.mlp.experts`, `model.layers.*.mlp.experts.act_fn` | 2640 |
| `d_head` | 256 | `model.layers.*.self_attn`, `model.layers.*.self_attn.q_norm`, `model.layers.*.self_attn.k_norm` | 2640 |
| `n_h` | 16 | `model.layers.*.self_attn`, `model.layers.*.self_attn.q_norm` | 2352 |
| `2*n_h*d_head` |  | `model.layers.*.linear_attn`, `model.layers.*.self_attn.q_proj`, `model.layers.*.linear_attn.conv1d`, `model.layers.*.self_attn` | 1944 |
| `n_kv` | 2 | `model.layers.*.self_attn`, `model.layers.*.self_attn.k_norm` | 1656 |
| `E` | 512 | `model.layers.*.mlp.gate`, `model.layers.*.mlp.experts` | 1536 |
| `n_v/n_k` |  | `model.layers.*.linear_attn` | 1224 |
| `n_h_lin_v*T` |  | `model.layers.*.linear_attn.norm`, `model.layers.*.linear_attn` | 1044 |
| `n_v*d_v` |  | `model.layers.*.linear_attn.out_proj`, `model.layers.*.linear_attn` | 936 |
| `d_conv_lin` | 4 | `model.layers.*.linear_attn`, `model.layers.*.linear_attn.conv1d` | 864 |
| `d_rope` |  | `model.layers.*.self_attn`, `model.rotary_emb` | 794 |
| `2*d_moe` |  | `model.layers.*.mlp.experts` | 672 |
| `2*n_k*d_k+2*n_v*d_v` |  | `model.layers.*.linear_attn.in_proj_qkvz`, `model.layers.*.linear_attn` | 648 |
| `(n_v/n_k)*d_v` |  | `model.layers.*.linear_attn` | 576 |
| `T+1` |  | `model.layers.*.self_attn` | 576 |
| `n_kv*d_head` |  | `model.layers.*.self_attn.k_proj`, `model.layers.*.self_attn.v_proj`, `model.layers.*.self_attn` | 480 |
| `n_h*d_head` |  | `model.layers.*.self_attn.o_proj`, `model.layers.*.self_attn` | 360 |
| `d_rope/2` |  | `model.layers.*.self_attn`, `model.rotary_emb` | 324 |
| `2*n_v` |  | `model.layers.*.linear_attn` | 324 |
| `n_h/n_kv` |  | `model.layers.*.self_attn` | 192 |
| `2*d_k+2*(n_v/n_k)*d_v` |  | `model.layers.*.linear_attn` | 144 |
| `d_head-d_rope` |  | `model.layers.*.self_attn` | 96 |
| `n_h+2*n_kv` |  | `model.layers.*.linear_attn.conv1d`, `model.layers.*.linear_attn` | 72 |
| `V` | 151936 | `lm_head`, `model.embed_tokens` | 20 |

### B. 이름 없이 남은 정수 전부 (63쌍)

**여기가 필터가 못 보던 자리다.** 정수가 남는 것 자체는 정상이다(루프 인덱스, 피연산자 개수, 브로드캐스트 축). 문제는 **이름이 있어야 하는데 없는 경우**이고, 마지막 열이 그 신호다 — 이 모델의 심볼과 값이 같다면 스코프가 그 모듈을 못 덮고 있을 수 있다. 실제로 `n_hc`(=4)가 그렇게 정수로 남아 있었다.

| 모듈 | 정수 | 축 수 | 같은 값의 심볼 |
|---|---|---|---|
| `model.layers.*.linear_attn` | 64 | 4536 | `d_chunk` |
| `model.layers.*.linear_attn` | 5 | 1116 | — |
| `model.layers.*.linear_attn` | 2 | 1008 | `n_kv` |
| `model.layers.*.linear_attn` | 3 | 1008 | — |
| `model.layers.*.linear_attn` | 4 | 1008 | `d_conv_lin` |
| `model.layers.*.linear_attn` | 6 | 1008 | — |
| `model.layers.*.linear_attn` | 7 | 1008 | — |
| `model.layers.*.linear_attn` | 8 | 1008 | — |
| `model.layers.*.linear_attn` | 9 | 1008 | — |
| `model.layers.*.linear_attn` | 10 | 1008 | `k` |
| `model.layers.*.linear_attn` | 11 | 1008 | — |
| `model.layers.*.linear_attn` | 12 | 1008 | — |
| `model.layers.*.linear_attn` | 13 | 1008 | — |
| `model.layers.*.linear_attn` | 14 | 1008 | — |
| `model.layers.*.linear_attn` | 15 | 1008 | — |
| `model.layers.*.linear_attn` | 16 | 1008 | `n_h`, `n_h_lin_k` |
| `model.layers.*.linear_attn` | 17 | 1008 | — |
| `model.layers.*.linear_attn` | 18 | 1008 | — |
| `model.layers.*.linear_attn` | 19 | 1008 | — |
| `model.layers.*.linear_attn` | 20 | 1008 | — |
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
| `model.layers.*.linear_attn` | 32 | 1008 | `n_h_lin_v` |
| `model.layers.*.linear_attn` | 33 | 1008 | — |
| `model.layers.*.linear_attn` | 34 | 1008 | — |
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
| `model.layers.*.linear_attn` | 48 | 1008 | `L` |
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

### C. 모듈이 내는 출력 shape 전부 (78개 모듈 / 623종)

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
  - `[[B, 1, d_model], [B, 1, d_model], [B, 1, n_v*d_v]]`
  - `[[B, 1, d_model]]`
  - `[[B, 1, n_h_lin_k, (n_v/n_k)*d_v]]`
  - `[[B, 1, n_h_lin_k, 1, d_head_lin_k]]`
  - `[[B, 1, n_h_lin_k, 2*d_k+2*(n_v/n_k)*d_v]]`
  - `[[B, 1, n_h_lin_k, d_conv_lin]]`
  - `[[B, 1, n_h_lin_k, d_head_lin_k], [B, 1, n_h_lin_k, d_head_lin_k], [B, 1, n_h_lin_k, (n_v/n_k)*d_v], [B, 1, n_h_lin_k, (n_v/n_k)*d_v]]`
  - `[[B, 1, n_h_lin_k, d_head_lin_k]]`
  - `[[B, 1, n_h_lin_k, n_v/n_k, d_head_lin_k]]`
  - `[[B, 1, n_h_lin_k, n_v/n_k], [B, 1, n_h_lin_k, n_v/n_k]]`
  - `[[B, 1, n_h_lin_k, n_v/n_k]]`
  - `[[B, 1, n_h_lin_v, 1]]`
  - `[[B, 1, n_h_lin_v, d_head_lin_k]]`
  - `[[B, 1, n_h_lin_v, d_head_lin_v]]`
  - `[[B, 1, n_h_lin_v]]`
  - `[[B, 1, n_v*d_v]]`
  - `[[B, 2*n_h*d_head, 1]]`
  - `[[B, 2*n_h*d_head, 5]]`
  - `[[B, 2*n_h*d_head, T]]`
  - `[[B, 2*n_h*d_head, d_conv_lin]]`
  - `[[B, 2*n_h*d_head, n_v/n_k]]`
  - `[[B, T, 2*n_h*d_head]]`
  - `[[B, T, d_model], [B, T, d_model], [B, T, n_v*d_v]]`
  - `[[B, T, d_model]]`
  - `[[B, T, n_h_lin_k, (n_v/n_k)*d_v]]`
  - `[[B, T, n_h_lin_k, 1, d_head_lin_k]]`
  - `[[B, T, n_h_lin_k, 2*d_k+2*(n_v/n_k)*d_v]]`
  - `[[B, T, n_h_lin_k, d_conv_lin]]`
  - `[[B, T, n_h_lin_k, d_head_lin_k], [B, T, n_h_lin_k, d_head_lin_k], [B, T, n_h_lin_k, (n_v/n_k)*d_v], [B, T, n_h_lin_k, (n_v/n_k)*d_v]]`
  - `[[B, T, n_h_lin_k, d_head_lin_k]]`
  - `[[B, T, n_h_lin_k, n_v/n_k, d_head_lin_k]]`
  - `[[B, T, n_h_lin_k, n_v/n_k], [B, T, n_h_lin_k, n_v/n_k]]`
  - `[[B, T, n_h_lin_k, n_v/n_k]]`
  - `[[B, T, n_h_lin_v, 1]]`
  - `[[B, T, n_h_lin_v, d_head_lin_k]]`
  - `[[B, T, n_h_lin_v, d_head_lin_v]]`
  - `[[B, T, n_h_lin_v]]`
  - `[[B, T, n_v*d_v]]`
  - `[[B, n_h_lin_v, 1, 1, 1]]`
  - `[[B, n_h_lin_v, 1, 1, 64]]`
  - `[[B, n_h_lin_v, 1, 1, d_chunk]]`
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
  - `[[B, n_h_lin_v, 1, d_chunk, 1]]`
  - `[[B, n_h_lin_v, 1, d_chunk, d_chunk]]`
  - `[[B, n_h_lin_v, 1, d_chunk, d_head_lin_k]]`
  - `[[B, n_h_lin_v, 1, d_chunk, d_head_lin_v]]`
  - `[[B, n_h_lin_v, 1, d_chunk]]`
  - `[[B, n_h_lin_v, 1, d_head_lin_k, d_chunk]]`
  - `[[B, n_h_lin_v, 1, d_head_lin_k]]`
  - `[[B, n_h_lin_v, 1, d_head_lin_v]]`
  - `[[B, n_h_lin_v, 1]]`
  - `[[B, n_h_lin_v, T, d_head_lin_k]]`
  - `[[B, n_h_lin_v, T, d_head_lin_v]]`
  - `[[B, n_h_lin_v, T]]`
  - `[[B, n_h_lin_v, d_chunk, 1]]`
  - `[[B, n_h_lin_v, d_chunk, d_chunk]]`
  - `[[B, n_h_lin_v, d_chunk, d_head_lin_k]]`
  - `[[B, n_h_lin_v, d_chunk, d_head_lin_v]]`
  - `[[B, n_h_lin_v, d_chunk]]`
  - `[[B, n_h_lin_v, d_head_lin_k, 1]]`
  - `[[B, n_h_lin_v, d_head_lin_k, d_chunk]]`
  - `[[B, n_h_lin_v, d_head_lin_k, d_head_lin_v]]`
  - `[[B, n_h_lin_v, d_head_lin_k]]`
  - `[[B, n_h_lin_v, d_head_lin_v]]`
  - `[[B, n_h_lin_v]]`
  - `[[d_chunk, 2*n_v]]`
  - `[[n_h_lin_v*T, d_head_lin_v]]`
  - `[[n_h_lin_v, d_chunk, d_chunk]]`
  - `[[n_h_lin_v, d_chunk, d_head_lin_k]]`
  - `[[n_h_lin_v, d_chunk, d_head_lin_v]]`
  - `[[n_h_lin_v, d_head_lin_k, d_chunk]]`
  - `[[n_h_lin_v, d_head_lin_k, d_head_lin_v]]`
  - `[[n_h_lin_v, d_head_lin_v]]`
  - `[[n_h_lin_v]]`
- `model.layers.*.linear_attn.conv1d`
  - `[[B, 2*n_h*d_head, n_h+2*n_kv]]`
- `model.layers.*.linear_attn.in_proj_ba`
  - `[[B, 1, d_chunk]]`
  - `[[B, T, d_chunk]]`
  - `[[B, d_chunk]]`
  - `[[B, d_model]]`
  - `[[T, d_chunk]]`
  - `[[T, d_model]]`
  - `[[d_model, d_chunk]]`
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
  - `[[n_h_lin_v*T, d_head_lin_v]]`
  - `[[n_h_lin_v, B]]`
  - `[[n_h_lin_v, d_head_lin_v]]`
- `model.layers.*.linear_attn.out_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[B, n_v*d_v]]`
  - `[[T, d_model]]`
  - `[[T, n_v*d_v]]`
  - `[[n_v*d_v, d_model]]`
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
  - `[[d_model, 1]]`
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
  - `[[B, 1, n_h, d_head], [B, 1, n_h, d_head]]`
  - `[[B, 1, n_h, d_head]]`
  - `[[B, 1, n_h, n_kv*d_head]]`
  - `[[B, 1, n_kv, d_head]]`
  - `[[B, T, n_h*d_head]]`
  - `[[B, T, n_h, d_head], [B, T, n_h, d_head]]`
  - `[[B, T, n_h, d_head]]`
  - `[[B, T, n_h, n_kv*d_head]]`
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
