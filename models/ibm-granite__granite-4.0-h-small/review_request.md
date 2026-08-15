# 검토 의뢰서 — ibm-granite/granite-4.0-h-small

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `granitemoehybrid`
- 판단 필요: **5건**

## 증거 — 이미 받아둔 실제 소스

- `develop/sources/modeling_granitemoehybrid.py` — 있음, 이 파일을 열어서 판정한다
- `develop/sources/configuration_granitemoehybrid.py` — 있음, 이 파일을 열어서 판정한다

- 온라인 원본: https://github.com/huggingface/transformers/tree/main/src/transformers/models/granitemoehybrid

그 밖의 재료: `full/review.md`(리뷰 패킷 — shape 별 실제 행 표본), `structure.yaml`(이 모델의 심볼 표), `full/<phase>.csv`(전체 operator 표).

## 판단이 필요한 것

### 2. 이 정사각 축이 정말 같은 이름 두 번인가

`[..., X, X]` 로 렌더됐는데, 그 이름이 읽은 config 필드에서 나온 정사각 reshape 을 modeling 소스에서 찾지 못했다. 두 축 크기가 우연히 같은 것일 수 있다.

- `d_state`
- `n_h_ssm`

### 3. 이름 붙일 근거가 없는 config 필드

모듈 폭으로 쓰이는데 심볼 표에 등록돼 있지 않다. 소스에서 무엇인지 확인하고 `rules/symbols.yaml` 에 등록하면 다음 실행부터 자동으로 잡힌다.

- `{'field': 'logits_scaling', 'value': 16, 'modules': 1}`
- `{'field': 'embedding_multiplier', 'value': 12, 'modules': 1}`

### 6. 값이 겹쳐 **임의로** 고른 축

두 심볼이 같은 값을 갖는 자리다. 규칙에는 고를 근거가 없고, 이긴 쪽은 전역 우선순위 — 즉 **관례**로 정해졌다. 이름이 맞을 수도 있지만 파이프라인은 그걸 알지 못한다. 표에서는 확신 있는 라벨과 똑같이 보인다.

**소스를 열어 어느 쪽인지 확정하는 것이 여기서 할 일이다.** 확정되면 `rules/label_overrides.yaml` 에 근거와 함께 못 박는다(review/05-overrides.md). 출신으로만 구별되는 경우라면 그렇게 적고 `open` 으로 남긴다.

- `d_state vs n_h_ssm` in `model.layers.*.mamba` — 값 128 를 두고 후보가 2개, 11556축

### 0. 규칙이 끝내지 못한 축 — **여기부터 답한다**

값으로는 결정할 수 없어 파이프라인이 판단을 넘긴 자리다. 세 가지뿐이다:
`tie`(두 심볼이 같은 값이라 관례로 골랐다) · `heur`(등록 규칙이 없어 산술로 지어냈다) · `bare`(이름을 못 붙였는데 크기가 커서 진짜 차원일 수 있다).

**답이 나오면 `override_stub` 을 채워 `rules/label_overrides.yaml` 에 넣는다.** `spread: class` 라 그 축이 지나는 모든 자리가 한 번에 바뀐다 — 모듈 경계에서 멈추지 않는다(그것이 예전에 교정을 막던 유일한 이유였다).

**값이 같은 심볼이 여럿이면 값으로는 영원히 못 가른다. shape 안의 위치가 말해 준다** — `[B, n_h, T, d_head]` 의 축 1 은 head 개수, 축 3 은 head 폭이다.

아래 `shape` 과 `축` 은 **그 축을 처음 만든 자리(앵커)** 의 것이다. 초안은 `shape`/`axis`/`field`/`shape_index`/`op_type`/`nth` 여섯으로 그 앵커를 지목한다 — `shape`+`axis` 만으로는 부족하다(Kimi 의 `[B, n_h, T, d_nope]` 축 3 은 **366개 등가류**에 걸쳐 있다: q 의 q_pass, KV 의 k_nope, value_states …). `nth` 는 그 모듈 안에서 같은 op_type 의 몇 번째인지다 — MLA 는 `self_attn` 안에 `split_with_sizes` 가 q용·kv용 둘이라 그것 없이는 못 가른다.

**유일성은 실제로 돌려 봐서 검증한다**: 그 조건에 맞는 자리들이 몇 개의 등가류에 속하는지 세고, **한 레이어 안에서 둘 이상**이면 `stub_ambiguous` 를 붙인다. 그 초안은 쓰지 말고 `open` 으로 남길 것.

| 왜 | 모듈 | 크기 | 지금 이름 | 후보 | 축 | 앵커 shape | 축 수 |
|---|---|---|---|---|---|---|---|
| `tie` | `model.layers.*.mamba` | 128 | `n_h_ssm` | `d_state`, `n_h_ssm` | 3 | `[B, d_state, d_head_ssm, n_h_ssm]` | 792 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 1 | `[B, d_state, d_head_ssm, n_h_ssm]` | 720 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 2 | `[B, d_chunk, d_state, d_head_ssm]` | 576 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 1 | `[B, d_state, 1, d_chunk]` | 504 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 2 | `[B, T, d_state]` | 396 |
| `tie` | `model.layers.*.mamba` | 128 | `n_h_ssm` | `d_state`, `n_h_ssm` | 3 | `[B, d_chunk, d_state, n_h_ssm]` | 396 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 1 | `[B, d_state, 1, d_chunk, d_chunk]` | 360 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 1 | `[B, d_state, 2, 2]` | 360 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 2 | `[B, d_chunk, d_state, n_h_ssm]` | 324 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 1 | `[B, d_state, d_head_ssm]` | 324 |
| `tie` | `model.layers.*.mamba` | 128 | `n_h_ssm` | `d_state`, `n_h_ssm` | 3 | `[B, 1, d_state, n_h_ssm]` | 288 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 0 | `[d_state]` | 252 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 2 | `[B, T, d_state, d_head_ssm]` | 252 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 3 | `[B, T, 1, d_state, n_h_ssm]` | 216 |
| `tie` | `model.layers.*.mamba` | 128 | `n_h_ssm` | `d_state`, `n_h_ssm` | 4 | `[B, T, 1, d_state, n_h_ssm]` | 216 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 4 | `[B, 1, d_chunk, d_chunk, d_state, 1]` | 216 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 2 | `[B, 1, d_state, d_head_ssm, n_h_ssm]` | 216 |
| `tie` | `model.layers.*.mamba` | 128 | `n_h_ssm` | `d_state`, `n_h_ssm` | 4 | `[B, 1, d_state, d_head_ssm, n_h_ssm]` | 216 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 2 | `[B, 1, d_state, n_h_ssm]` | 216 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 1 | `[B, d_state]` | 180 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 2 | `[B, d_chunk, d_state]` | 144 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 2 | `[B, 2, d_state, d_head_ssm, n_h_ssm]` | 108 |
| `tie` | `model.layers.*.mamba` | 128 | `n_h_ssm` | `d_state`, `n_h_ssm` | 4 | `[B, 2, d_state, d_head_ssm, n_h_ssm]` | 108 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 0 | `[d_state, B]` | 72 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 2 | `[B, T, d_state, 1]` | 72 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 1 | `[B, d_state, 1, d_chunk, 1]` | 72 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 4 | `[B, 1, d_chunk, d_chunk, d_state, n_h_ssm]` | 72 |
| `tie` | `model.layers.*.mamba` | 128 | `n_h_ssm` | `d_state`, `n_h_ssm` | 5 | `[B, 1, d_chunk, d_chunk, d_state, n_h_ssm]` | 72 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 4 | `[B, 1, d_chunk, d_chunk, d_state]` | 72 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 4 | `[B, 1, d_chunk, d_chunk, d_state, d_head_ssm]` | 72 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 1 | `[B, d_state, 1, 1]` | 72 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 3 | `[B, 1, d_chunk, d_state]` | 72 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 3 | `[B, 1, d_chunk, d_state, 1]` | 72 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 3 | `[B, 1, d_chunk, d_state, 1, n_h_ssm]` | 72 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 3 | `[B, 1, d_chunk, d_state, d_head_ssm, 1]` | 72 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 3 | `[B, 1, d_chunk, d_state, d_head_ssm, n_h_ssm]` | 72 |
| `tie` | `model.layers.*.mamba` | 128 | `n_h_ssm` | `d_state`, `n_h_ssm` | 5 | `[B, 1, d_chunk, d_state, d_head_ssm, n_h_ssm]` | 72 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 1 | `[B, d_state, 1]` | 72 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 1 | `[B, d_state, 2]` | 72 |
| `tie` | `model.layers.*.mamba` | 128 | `d_state` | `d_state`, `n_h_ssm` | 1 | `[B, d_state, 2, 1]` | 72 |

**고칠 것과 맞는 것 둘 다 적는다.** 이름이 틀렸으면 아래 초안의 `to`/`source` 를 채워 `rules/label_overrides.yaml` 에, **지금 이름이 맞으면** 같은 앵커에 `to` 대신 `label: <지금 이름>` 과 `source` 를 적어 `rules/label_confirmed.yaml` 에 넣는다. 확인을 적지 않으면 그 축은 재생성마다 다시 질문으로 올라온다.

초안(그대로 복사해 `to` 와 `source` 만 채운다):

```yaml
  - model: ibm-granite__granite-4.0-h-small
    module: 'mamba$'
    spread: class
    shape: ["B", "d_state", "d_head_ssm", "n_h_ssm"]
    axis: 3
    field: o
    shape_index: 0
    op_type: elementwise_mul
    nth: 1
    from: n_h_ssm
    to: <소스가 말하는 이름>
    expect: 128
    source: <modeling_*.py:줄 인용>
  - model: ibm-granite__granite-4.0-h-small
    module: 'mamba$'
    spread: class
    shape: ["B", "d_state", "d_head_ssm", "n_h_ssm"]
    axis: 1
    field: o
    shape_index: 0
    op_type: elementwise_mul
    nth: 1
    from: d_state
    to: <소스가 말하는 이름>
    expect: 128
    source: <modeling_*.py:줄 인용>
  - model: ibm-granite__granite-4.0-h-small
    module: 'mamba$'
    spread: class
    shape: ["B", "d_chunk", "d_state", "d_head_ssm"]
    axis: 2
    field: o
    shape_index: 0
    op_type: constant_pad_nd
    nth: 1
    from: d_state
    to: <소스가 말하는 이름>
    expect: 128
    source: <modeling_*.py:줄 인용>
  - model: ibm-granite__granite-4.0-h-small
    module: 'mamba$'
    spread: class
    shape: ["B", "d_state", "1", "d_chunk"]
    axis: 1
    field: o
    shape_index: 0
    op_type: permute
    nth: 0
    from: d_state
    to: <소스가 말하는 이름>
    expect: 128
    source: <modeling_*.py:줄 인용>
  - model: ibm-granite__granite-4.0-h-small
    module: 'mamba$'
    spread: class
    shape: ["B", "T", "d_state"]
    axis: 2
    field: o
    shape_index: 2
    op_type: split_with_sizes
    nth: 0
    from: d_state
    to: <소스가 말하는 이름>
    expect: 128
    source: <modeling_*.py:줄 인용>
  - model: ibm-granite__granite-4.0-h-small
    module: 'mamba$'
    spread: class
    shape: ["B", "d_chunk", "d_state", "n_h_ssm"]
    axis: 3
    field: o
    shape_index: 0
    op_type: constant_pad_nd
    nth: 4
    from: n_h_ssm
    to: <소스가 말하는 이름>
    expect: 128
    source: <modeling_*.py:줄 인용>
```

## 기계적으로 이미 확인된 것 — 다시 묻지 말 것

- **심볼이 읽은 config 필드**: 전부 이 모델의 config 클래스(또는 상속/프로퍼티/getattr 기본값)에 존재한다
- **정사각 축**: `d_chunk` ← 소스의 `chunk_size` ← `mamba_chunk_size`
- **모듈이 읽는 config 속성**: `__init__` 에서 config 를 읽는 클래스 10개를 소스에서 확인했다. 그 목록이 각 모듈의 폭이 가질 수 있는 이름의 전부다.
- **가중치 축 ↔ 모듈 소속**: 가중치 축의 이름이 전부 그 모듈(또는 그 부모)이 실제로 읽는 config 필드에서 나왔다. 이 축들은 값이 아니라 소스로 확인된 것이다.

## 행 단위 전건 — 여기부터 읽는다

접힌 표(`<phase>.jsonl`)의 **고유 행 전부**다. 레이어 인덱스만 접었고 그 밖에는 아무것도 합치지 않았다. 아래 A/B/C 절은 (모듈, 라벨)로 접은 뷰라 **한 행 안의 어긋남이 보이지 않는다** — 실제로 외부 검토가 찾아낸 결함 세 건이 전부 그 자리에 있었다.

**한 행씩 읽고 이것만 물어라: 입력·가중치·출력이 서로 말이 되는가.**

- 같은 텐서가 `input_shape` 와 `weight_shape` 에서 다른 이름을 쓰지 않는가 (가중치는 `[out, in]` 으로 저장되고 피연산자는 전치돼 있다)
- 전치·view 처럼 **이름을 바꿀 수 없는 op** 이 이름을 바꾸지 않았는가
- 행렬곱의 수축 축이 양쪽에서 같은 이름인가 — `[m,k] @ [k,n] -> [m,n]`
- 이 모듈이 그 이름을 가질 수 있는가 (소스에서 그 `nn.Linear` 를 만드는 줄을 찾아라)

고유 행 68개.

| phase | 모듈 | op | input_shape | weight_shape | output_shape |
|---|---|---|---|---|---|
| prefill | `model.embed_tokens` | embedding | `[['V', 'd_model'], ['B', 'T']]` | `['V', 'd_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model` | elementwise_mul | `[['B', 'T', 'd_model']]` | `None` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.input_layernorm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.mamba.in_proj` | matmul | `[['T', 'd_model'], ['d_model', '2*d_inner+2*n_g*d_state+n_h_ssm']]` | `['2*d_inner+2*n_g*d_state+n_h_ssm', 'd_model']` | `[['T', '2*d_inner+2*n_g*d_state+n_h_ssm']]` |
| prefill | `model.layers.*.mamba.conv1d` | conv1d | `[['B', 'd_inner+2*n_g*d_state', 'T'], ['d_inner+2*n_g*d_state', '1', 'd_conv'], ['d_inner+2*n_g*d_state']]` | `['d_inner+2*n_g*d_state', '1', 'd_conv']` | `[['B', 'd_inner+2*n_g*d_state', 'T+d_conv-1']]` |
| prefill | `model.layers.*.mamba.act` | silu | `[['B', 'd_inner+2*n_g*d_state', 'T']]` | `None` | `[['B', 'd_inner+2*n_g*d_state', 'T']]` |
| prefill | `model.layers.*.mamba` | exp | `[['d_state']]` | `None` | `[['d_state']]` |
| prefill | `model.layers.*.mamba` | exp | `[['B', 'd_state', '1', 'd_chunk', 'd_chunk']]` | `None` | `[['B', 'd_state', '1', 'd_chunk', 'd_chunk']]` |
| prefill | `model.layers.*.mamba` | exp | `[['B', 'd_state', '1', 'd_chunk']]` | `None` | `[['B', 'd_state', '1', 'd_chunk']]` |
| prefill | `model.layers.*.mamba` | exp | `[['B', 'd_state', '2', '2']]` | `None` | `[['B', 'd_state', '2', '2']]` |
| prefill | `model.layers.*.mamba.norm` | rmsnorm | `[['B', 'T', 'd_inner']]` | `['d_inner']` | `[['B', 'T', 'd_inner']]` |
| prefill | `model.layers.*.mamba.out_proj` | matmul | `[['T', 'd_inner'], ['d_inner', 'd_model']]` | `['d_model', 'd_inner']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_mul | `[['B', 'T', 'd_model']]` | `None` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_add | `[['B', 'T', 'd_model'], ['B', 'T', 'd_model']]` | `None` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.post_attention_layernorm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.block_sparse_moe.router` | matmul | `[['T', 'd_model'], ['d_model', 'E']]` | `['E', 'd_model']` | `[['T', 'E']]` |
| prefill | `model.layers.*.block_sparse_moe.router` | softmax | `[['T', 'k']]` | `None` | `[['T', 'k']]` |
| prefill | `model.layers.*.block_sparse_moe.experts` | grouped_matmul | `[['k*T', 'd_model'], ['E', 'd_model', '2*d_moe'], ['E']]` | `['E', '2*d_moe', 'd_model']` | `[['k*T', '2*d_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.experts.act_fn` | silu | `[['k*T', 'd_moe']]` | `None` | `[['k*T', 'd_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.experts` | elementwise_mul | `[['k*T', 'd_moe'], ['k*T', 'd_moe']]` | `None` | `[['k*T', 'd_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.experts` | grouped_matmul | `[['k*T', 'd_moe'], ['E', 'd_moe', 'd_model'], ['E']]` | `['E', 'd_model', 'd_moe']` | `[['k*T', 'd_model']]` |
| prefill | `model.layers.*.block_sparse_moe.experts` | elementwise_mul | `[['k*T', 'd_model'], ['k*T', 'B']]` | `None` | `[['k*T', 'd_model']]` |
| prefill | `model.layers.*.block_sparse_moe.experts` | sum | `[['T', 'k', 'd_model']]` | `None` | `[['T', 'd_model']]` |
| prefill | `model.layers.*.shared_mlp.input_linear` | matmul | `[['T', 'd_model'], ['d_model', '2*d_shared']]` | `['2*d_shared', 'd_model']` | `[['T', '2*d_shared']]` |
| prefill | `model.layers.*.shared_mlp.activation` | silu | `[['B', 'T', 'd_shared']]` | `None` | `[['B', 'T', 'd_shared']]` |
| prefill | `model.layers.*.shared_mlp.output_linear` | matmul | `[['T', 'd_shared'], ['d_shared', 'd_model']]` | `['d_model', 'd_shared']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*.self_attn.q_proj` | matmul | `[['T', 'd_model'], ['d_model', 'n_h*d_head']]` | `['n_h*d_head', 'd_model']` | `[['T', 'n_h*d_head']]` |
| prefill | `model.layers.*.self_attn.k_proj` | matmul | `[['T', 'd_model'], ['d_model', 'n_kv*d_head']]` | `['n_kv*d_head', 'd_model']` | `[['T', 'n_kv*d_head']]` |
| prefill | `model.layers.*.self_attn.v_proj` | matmul | `[['T', 'd_model'], ['d_model', 'n_kv*d_head']]` | `['n_kv*d_head', 'd_model']` | `[['T', 'n_kv*d_head']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'T', 'd_head'], ['n_h', 'd_head', 'T']]` | `None` | `[['n_h', 'T', 'T']]` |
| prefill | `model.layers.*.self_attn` | softmax | `[['B', 'n_h', 'T', 'T']]` | `None` | `[['B', 'n_h', 'T', 'T']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'T', 'T'], ['n_h', 'T', 'd_head']]` | `None` | `[['n_h', 'T', 'd_head']]` |
| prefill | `model.layers.*.self_attn.o_proj` | matmul | `[['T', 'n_h*d_head'], ['n_h*d_head', 'd_model']]` | `['d_model', 'n_h*d_head']` | `[['T', 'd_model']]` |
| prefill | `model.norm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `lm_head` | matmul | `[['T', 'd_model'], ['d_model', 'V']]` | `['V', 'd_model']` | `[['T', 'V']]` |
| decode | `model.embed_tokens` | embedding | `[['V', 'd_model'], ['B', '1']]` | `['V', 'd_model']` | `[['B', '1', 'd_model']]` |
| decode | `model` | elementwise_mul | `[['B', '1', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.input_layernorm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.mamba.in_proj` | matmul | `[['B', 'd_model'], ['d_model', '2*d_inner+2*n_g*d_state+n_h_ssm']]` | `['2*d_inner+2*n_g*d_state+n_h_ssm', 'd_model']` | `[['B', '2*d_inner+2*n_g*d_state+n_h_ssm']]` |
| decode | `model.layers.*.mamba.act` | silu | `[['B', 'd_inner+2*n_g*d_state']]` | `None` | `[['B', 'd_inner+2*n_g*d_state']]` |
| decode | `model.layers.*.mamba` | exp | `[['d_state']]` | `None` | `[['d_state']]` |
| decode | `model.layers.*.mamba` | exp | `[['B', 'd_state', 'd_head_ssm', 'n_h_ssm']]` | `None` | `[['B', 'd_state', 'd_head_ssm', 'n_h_ssm']]` |
| decode | `model.layers.*.mamba` | batched_matmul | `[['d_state', 'd_head_ssm', 'n_h_ssm'], ['d_state', 'n_h_ssm', 'B']]` | `None` | `[['d_state', 'd_head_ssm', 'B']]` |
| decode | `model.layers.*.mamba.norm` | rmsnorm | `[['B', '1', 'd_inner']]` | `['d_inner']` | `[['B', '1', 'd_inner']]` |
| decode | `model.layers.*.mamba.out_proj` | matmul | `[['B', 'd_inner'], ['d_inner', 'd_model']]` | `['d_model', 'd_inner']` | `[['B', 'd_model']]` |
| decode | `model.layers.*` | elementwise_mul | `[['B', '1', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*` | elementwise_add | `[['B', '1', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.post_attention_layernorm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.block_sparse_moe.router` | matmul | `[['B', 'd_model'], ['d_model', 'E']]` | `['E', 'd_model']` | `[['B', 'E']]` |
| decode | `model.layers.*.block_sparse_moe.router` | softmax | `[['B', 'k']]` | `None` | `[['B', 'k']]` |
| decode | `model.layers.*.block_sparse_moe.experts` | grouped_matmul | `[['k', 'd_model'], ['E', 'd_model', '2*d_moe'], ['E']]` | `['E', '2*d_moe', 'd_model']` | `[['k', '2*d_moe']]` |
| decode | `model.layers.*.block_sparse_moe.experts.act_fn` | silu | `[['k', 'd_moe']]` | `None` | `[['k', 'd_moe']]` |
| decode | `model.layers.*.block_sparse_moe.experts` | elementwise_mul | `[['k', 'd_moe'], ['k', 'd_moe']]` | `None` | `[['k', 'd_moe']]` |
| decode | `model.layers.*.block_sparse_moe.experts` | grouped_matmul | `[['k', 'd_moe'], ['E', 'd_moe', 'd_model'], ['E']]` | `['E', 'd_model', 'd_moe']` | `[['k', 'd_model']]` |
| decode | `model.layers.*.block_sparse_moe.experts` | elementwise_mul | `[['k', 'd_model'], ['k', 'B']]` | `None` | `[['k', 'd_model']]` |
| decode | `model.layers.*.block_sparse_moe.experts` | sum | `[['B', 'k', 'd_model']]` | `None` | `[['B', 'd_model']]` |
| decode | `model.layers.*.shared_mlp.input_linear` | matmul | `[['B', 'd_model'], ['d_model', '2*d_shared']]` | `['2*d_shared', 'd_model']` | `[['B', '2*d_shared']]` |
| decode | `model.layers.*.shared_mlp.activation` | silu | `[['B', '1', 'd_shared']]` | `None` | `[['B', '1', 'd_shared']]` |
| decode | `model.layers.*.shared_mlp.output_linear` | matmul | `[['B', 'd_shared'], ['d_shared', 'd_model']]` | `['d_model', 'd_shared']` | `[['B', 'd_model']]` |
| decode | `model.layers.*.self_attn.q_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_h*d_head']]` | `['n_h*d_head', 'd_model']` | `[['B', 'n_h*d_head']]` |
| decode | `model.layers.*.self_attn.k_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_kv*d_head']]` | `['n_kv*d_head', 'd_model']` | `[['B', 'n_kv*d_head']]` |
| decode | `model.layers.*.self_attn.v_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_kv*d_head']]` | `['n_kv*d_head', 'd_model']` | `[['B', 'n_kv*d_head']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'B', 'd_head'], ['n_h', 'd_head', 'T+1']]` | `None` | `[['n_h', 'B', 'T+1']]` |
| decode | `model.layers.*.self_attn` | softmax | `[['B', 'n_h', '1', 'T+1']]` | `None` | `[['B', 'n_h', '1', 'T+1']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'B', 'T+1'], ['n_h', 'T+1', 'd_head']]` | `None` | `[['n_h', 'B', 'd_head']]` |
| decode | `model.layers.*.self_attn.o_proj` | matmul | `[['B', 'n_h*d_head'], ['n_h*d_head', 'd_model']]` | `['d_model', 'n_h*d_head']` | `[['B', 'd_model']]` |
| decode | `model.norm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `lm_head` | matmul | `[['B', 'd_model'], ['d_model', 'V']]` | `['V', 'd_model']` | `[['B', 'V']]` |

## 전수 점검 — 이 모델이 쓰는 이름 전부

위 절이 '풀리지 않은 것'이라면 여기는 **전부**다. 규칙이 자신 있게 붙인 이름도 틀릴 수 있고, 그런 건 미결 목록에 절대 오르지 않는다. 한 줄씩 읽고 **그 모듈에서 그 이름이 말이 되는지** 보라.

### A. 붙은 이름 전부 (28종)

| 라벨 | 값 | 나타나는 모듈 | 축 수 |
|---|---|---|---|
| `B` |  | `model.layers.*.mamba`, `model.layers.*.mamba.norm`, `model.layers.*.input_layernorm`, `model.layers.*.post_attention_layernorm` 외 61개 | 21960 |
| `d_state` | 128 | `model.layers.*.mamba` | 11556 |
| `d_model` | 4096 | `model.layers.*.block_sparse_moe.experts`, `model.layers.*.input_layernorm`, `model.layers.*.post_attention_layernorm`, `model.layers.*.shared_mlp.input_linear` 외 54개 | 8582 |
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
| `n_kv` | 8 | `model.layers.*.self_attn` | 200 |
| `n_h*d_head` |  | `model.layers.*.self_attn.q_proj`, `model.layers.*.self_attn.o_proj`, `model.layers.*.self_attn` | 144 |
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

### C. 모듈이 내는 출력 shape 전부 (66개 모듈 / 324종)

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
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[B, n_h*d_head]]`
  - `[[T, d_model]]`
  - `[[T, n_h*d_head]]`
  - `[[n_h*d_head, d_model]]`
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
