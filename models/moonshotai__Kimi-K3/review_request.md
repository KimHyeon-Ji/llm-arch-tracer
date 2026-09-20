# 검토 의뢰서 — moonshotai/Kimi-K3

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `kimi_linear`
- 판단 필요: **6건**

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

### 6. 값이 겹쳐 **임의로** 고른 축

두 심볼이 같은 값을 갖는 자리다. 규칙에는 고를 근거가 없고, 이긴 쪽은 전역 우선순위 — 즉 **관례**로 정해졌다. 이름이 맞을 수도 있지만 파이프라인은 그걸 알지 못한다. 표에서는 확신 있는 라벨과 똑같이 보인다.

**소스를 열어 어느 쪽인지 확정하는 것이 여기서 할 일이다.** 확정되면 `rules/label_overrides.yaml` 에 근거와 함께 못 박는다(review/05-overrides.md). 출신으로만 구별되는 경우라면 그렇게 적고 `open` 으로 남긴다.

- `n_h vs n_kv` in `model.layers.*.self_attn` — 값 96 를 두고 후보가 2개, 3600축
- `d_nope vs d_v` in `model.layers.*.self_attn` — 값 128 를 두고 후보가 2개, 1320축

### 0. 규칙이 끝내지 못한 축 — **여기부터 답한다**

값으로는 결정할 수 없어 파이프라인이 판단을 넘긴 자리다. 세 가지뿐이다:
`tie`(두 심볼이 같은 값이라 관례로 골랐다) · `heur`(등록 규칙이 없어 산술로 지어냈다) · `bare`(이름을 못 붙였는데 크기가 커서 진짜 차원일 수 있다).

**답이 나오면 `override_stub` 을 채워 `rules/label_overrides.yaml` 에 넣는다.** `spread: class` 라 그 축이 지나는 모든 자리가 한 번에 바뀐다 — 모듈 경계에서 멈추지 않는다(그것이 예전에 교정을 막던 유일한 이유였다).

**값이 같은 심볼이 여럿이면 값으로는 영원히 못 가른다. shape 안의 위치가 말해 준다** — `[B, n_h, T, d_head]` 의 축 1 은 head 개수, 축 3 은 head 폭이다.

아래 `shape` 과 `축` 은 **그 축을 처음 만든 자리(앵커)** 의 것이다. 초안은 `shape`/`axis`/`field`/`shape_index`/`op_type`/`nth` 여섯으로 그 앵커를 지목한다 — `shape`+`axis` 만으로는 부족하다(Kimi 의 `[B, n_h, T, d_nope]` 축 3 은 **366개 등가류**에 걸쳐 있다: q 의 q_pass, KV 의 k_nope, value_states …). `nth` 는 그 모듈 안에서 같은 op_type 의 몇 번째인지다 — MLA 는 `self_attn` 안에 `split_with_sizes` 가 q용·kv용 둘이라 그것 없이는 못 가른다.

**유일성은 실제로 돌려 봐서 검증한다**: 그 조건에 맞는 자리들이 몇 개의 등가류에 속하는지 세고, **한 레이어 안에서 둘 이상**이면 `stub_ambiguous` 를 붙인다. 그 초안은 쓰지 말고 `open` 으로 남길 것.

| 왜 | 모듈 | 크기 | 지금 이름 | 후보 | 축 | 앵커 shape | 축 수 |
|---|---|---|---|---|---|---|---|
| `bare` | `model.layers.*.block_sparse_moe.experts.*.act_fn` | 3840 | `3840` | — | 0 | `[3840, d_moe]` | 11040 |
| ⚠ | `model.layers.*.block_sparse_moe.experts.*.act_fn` | | | | | **이 초안은 한 레이어 안에서 등가류 4개를 동시에 잡는다 — 그대로 쓰면 나머지가 망가진다. 이 축은 위치 선택자로 지목할 수 없으니 `open` 으로 남길 것.** | |
| `bare` | `model.layers.*.block_sparse_moe.experts.*` | 3840 | `3840` | — | 0 | `[3840, d_moe]` | 1840 |
| ⚠ | `model.layers.*.block_sparse_moe.experts.*` | | | | | **이 초안은 한 레이어 안에서 등가류 4개를 동시에 잡는다 — 그대로 쓰면 나머지가 망가진다. 이 축은 위치 선택자로 지목할 수 없으니 `open` 으로 남길 것.** | |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h_kda` | `n_h`, `n_kv` | 2 | `[B, 1, n_h_kda, d_head_kda]` | 1380 |
| `tie` | `model.layers.*.self_attn` | 128 | `d_head_kda` | `d_nope`, `d_v` | 3 | `[B, 1, n_h_kda, d_head_kda]` | 1380 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h_kda` | `n_h`, `n_kv` | 1 | `[B, n_h_kda, 5, 4]` | 759 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h_kda` | `n_h`, `n_kv` | 1 | `[B, n_h_kda, 5, 8]` | 759 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h_kda` | `n_h`, `n_kv` | 1 | `[B, n_h_kda, 5, 10]` | 759 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h_kda` | `n_h`, `n_kv` | 1 | `[B, n_h_kda, 5, 12]` | 759 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h_kda` | `n_h`, `n_kv` | 1 | `[B, n_h_kda, 5, 16]` | 759 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h_kda` | `n_h`, `n_kv` | 1 | `[B, n_h_kda, 5, 37]` | 759 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h_kda` | `n_h`, `n_kv` | 1 | `[B, n_h_kda, 5, 48]` | 759 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h_kda` | `n_h`, `n_kv` | 1 | `[B, n_h_kda, 5, d_chunk, d_chunk]` | 690 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h_kda` | `n_h`, `n_kv` | 1 | `[B, n_h_kda, d_chunk, d_head_kda]` | 621 |
| `tie` | `model.layers.*.self_attn` | 128 | `d_head_kda` | `d_nope`, `d_v` | 4 | `[B, n_h_kda, 5, d_chunk, d_head_kda]` | 552 |
| `tie` | `model.layers.*.self_attn` | 128 | `d_head_kda` | `d_nope`, `d_v` | 3 | `[B, n_h_kda, d_chunk, d_head_kda]` | 552 |
| `tie` | `model.layers.*.self_attn` | 128 | `d_head_kda` | `d_nope`, `d_v` | 3 | `[B, n_h_kda, d_head_kda, d_head_kda]` | 552 |
| `bare` | `model.layers.*.block_sparse_moe` | 3840 | `3840` | — | 0 | `[3840, d_moe_lat]` | 460 |
| `tie` | `model.layers.*.self_attn` | 128 | `d_head_kda` | `d_nope`, `d_v` | 2 | `[B*n_h_kda, 1, d_head_kda]` | 414 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h` | `n_h`, `n_kv` | 1 | `[B, n_h, T, T]` | 336 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h` | `n_h`, `n_kv` | 1 | `[B, n_h, 1, T+1]` | 336 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h_kda` | `n_h`, `n_kv` | 0 | `[n_h_kda, 1]` | 276 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h_kda` | `n_h`, `n_kv` | 1 | `[B, n_h_kda, d_head_kda]` | 276 |
| `tie` | `model.layers.*.self_attn` | 128 | `d_head_kda` | `d_nope`, `d_v` | 2 | `[B, n_h_kda, d_head_kda]` | 276 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h` | `n_h`, `n_kv` | 1 | `[B, n_h, 1, d_rope]` | 240 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h` | `n_h`, `n_kv` | 1 | `[B, n_h, T, d_nope]` | 192 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h` | `n_h`, `n_kv` | 1 | `[B, n_h, 1, d_nope]` | 192 |
| `tie` | `model.layers.*.self_attn` | 128 | `d_nope` | `d_nope`, `d_v` | 4 | `[B, n_h, T+1, 1, d_nope]` | 192 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h` | `n_h`, `n_kv` | 1 | `[B, n_h, T, d_rope]` | 168 |
| `tie` | `model.layers.*.self_attn` | 128 | `d_nope` | `d_nope`, `d_v` | 2 | `[B*n_h, T, d_nope]` | 144 |
| `tie` | `model.layers.*.self_attn` | 128 | `d_nope` | `d_nope`, `d_v` | 3 | `[B, n_h, T, d_nope]` | 144 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h_kda` | `n_h`, `n_kv` | 2 | `[B, T, n_h_kda, 1]` | 138 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h_kda` | `n_h`, `n_kv` | 1 | `[B, n_h_kda, 5, 1, d_head_kda]` | 138 |
| `tie` | `model.layers.*.self_attn` | 128 | `d_head_kda` | `d_nope`, `d_v` | 4 | `[B, n_h_kda, 5, 1, d_head_kda]` | 138 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h_kda` | `n_h`, `n_kv` | 1 | `[B, n_h_kda, 5, d_chunk, d_head_kda]` | 138 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h_kda` | `n_h`, `n_kv` | 1 | `[B, n_h_kda, 5, d_head_kda, 1]` | 138 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h_kda` | `n_h`, `n_kv` | 1 | `[B, n_h_kda, 5, d_chunk, 1]` | 138 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h_kda` | `n_h`, `n_kv` | 1 | `[B, n_h_kda, 5, d_chunk]` | 138 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h_kda` | `n_h`, `n_kv` | 1 | `[B, n_h_kda, 1, d_head_kda]` | 138 |
| `tie` | `model.layers.*.self_attn` | 128 | `d_head_kda` | `d_nope`, `d_v` | 3 | `[B, n_h_kda, 1, d_head_kda]` | 138 |
| `tie` | `model.layers.*.self_attn` | 96 | `n_h_kda` | `n_h`, `n_kv` | 1 | `[B, n_h_kda, d_head_kda, 1]` | 138 |

**고칠 것과 맞는 것 둘 다 적는다.** 이름이 틀렸으면 아래 초안의 `to`/`source` 를 채워 `rules/label_overrides.yaml` 에, **지금 이름이 맞으면** 같은 앵커에 `to` 대신 `label: <지금 이름>` 과 `source` 를 적어 `rules/label_confirmed.yaml` 에 넣는다. 확인을 적지 않으면 그 축은 재생성마다 다시 질문으로 올라온다.

초안(그대로 복사해 `to` 와 `source` 만 채운다):

```yaml
  - model: moonshotai__Kimi-K3
    module: '^model\.layers\.\*\.block_sparse_moe\.experts\.\*\.act_fn$'
    spread: class
    shape: ["3840", "d_moe"]
    axis: 0
    field: o
    shape_index: 0
    op_type: slice
    nth: 0
    from: 3840
    to: <소스가 말하는 이름>
    expect: 3840
    source: <modeling_*.py:줄 인용>
  - model: moonshotai__Kimi-K3
    module: '^model\.layers\.\*\.block_sparse_moe\.experts\.\*$'
    spread: class
    shape: ["3840", "d_moe"]
    axis: 0
    field: i
    shape_index: 0
    op_type: concat
    nth: 0
    from: 3840
    to: <소스가 말하는 이름>
    expect: 3840
    source: <modeling_*.py:줄 인용>
  - model: moonshotai__Kimi-K3
    module: 'self_attn$'
    spread: class
    shape: ["B", "1", "n_h_kda", "d_head_kda"]
    axis: 2
    field: i
    shape_index: 0
    op_type: _to_copy
    nth: 5
    from: n_h_kda
    to: <소스가 말하는 이름>
    expect: 96
    source: <modeling_*.py:줄 인용>
  - model: moonshotai__Kimi-K3
    module: 'self_attn$'
    spread: class
    shape: ["B", "1", "n_h_kda", "d_head_kda"]
    axis: 3
    field: i
    shape_index: 0
    op_type: _to_copy
    nth: 5
    from: d_head_kda
    to: <소스가 말하는 이름>
    expect: 128
    source: <modeling_*.py:줄 인용>
  - model: moonshotai__Kimi-K3
    module: 'self_attn$'
    spread: class
    shape: ["B", "n_h_kda", "5", "4"]
    axis: 1
    field: o
    shape_index: 0
    op_type: slice
    nth: 73
    from: n_h_kda
    to: <소스가 말하는 이름>
    expect: 96
    source: <modeling_*.py:줄 인용>
  - model: moonshotai__Kimi-K3
    module: 'self_attn$'
    spread: class
    shape: ["B", "n_h_kda", "5", "8"]
    axis: 1
    field: o
    shape_index: 0
    op_type: slice
    nth: 85
    from: n_h_kda
    to: <소스가 말하는 이름>
    expect: 96
    source: <modeling_*.py:줄 인용>
```

## 기계적으로 이미 확인된 것 — 다시 묻지 말 것

- **심볼이 읽은 config 필드**: 3건이 클래스 선언 밖 — 위 1절 참고
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

고유 행 263개.

| phase | 모듈 | op | input_shape | weight_shape | output_shape |
|---|---|---|---|---|---|
| prefill | `model.embed_tokens` | embedding | `[['V', 'd_model'], ['B', 'T']]` | `['V', 'd_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*` | concat | `[['B*T', '0', 'd_model'], ['B*T', '1', 'd_model']]` | `None` | `[['B*T', '1', 'd_model']]` |
| prefill | `model.layers.*.input_layernorm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.self_attn.q_proj` | matmul | `[['B*T', 'd_model'], ['d_model', 'n_h_kda*d_head_kda']]` | `['n_h_kda*d_head_kda', 'd_model']` | `[['B*T', 'n_h_kda*d_head_kda']]` |
| prefill | `model.layers.*.self_attn.k_proj` | matmul | `[['B*T', 'd_model'], ['d_model', 'n_h_kda*d_head_kda']]` | `['n_h_kda*d_head_kda', 'd_model']` | `[['B*T', 'n_h_kda*d_head_kda']]` |
| prefill | `model.layers.*.self_attn.v_proj` | matmul | `[['B*T', 'd_model'], ['d_model', 'n_h_kda*d_head_kda']]` | `['n_h_kda*d_head_kda', 'd_model']` | `[['B*T', 'n_h_kda*d_head_kda']]` |
| prefill | `model.layers.*.self_attn.q_conv1d.conv` | conv1d | `[['B', 'n_h_kda*d_head_kda', 'T'], ['n_h_kda*d_head_kda', '1', 'd_conv']]` | `['n_h_kda*d_head_kda', '1', 'd_conv']` | `[['B', 'n_h_kda*d_head_kda', 'T+d_conv-1']]` |
| prefill | `model.layers.*.self_attn.q_conv1d` | silu | `[['B', 'T', 'n_h_kda*d_head_kda']]` | `None` | `[['B', 'T', 'n_h_kda*d_head_kda']]` |
| prefill | `model.layers.*.self_attn.k_conv1d.conv` | conv1d | `[['B', 'n_h_kda*d_head_kda', 'T'], ['n_h_kda*d_head_kda', '1', 'd_conv']]` | `['n_h_kda*d_head_kda', '1', 'd_conv']` | `[['B', 'n_h_kda*d_head_kda', 'T+d_conv-1']]` |
| prefill | `model.layers.*.self_attn.k_conv1d` | silu | `[['B', 'T', 'n_h_kda*d_head_kda']]` | `None` | `[['B', 'T', 'n_h_kda*d_head_kda']]` |
| prefill | `model.layers.*.self_attn.v_conv1d.conv` | conv1d | `[['B', 'n_h_kda*d_head_kda', 'T'], ['n_h_kda*d_head_kda', '1', 'd_conv']]` | `['n_h_kda*d_head_kda', '1', 'd_conv']` | `[['B', 'n_h_kda*d_head_kda', 'T+d_conv-1']]` |
| prefill | `model.layers.*.self_attn.v_conv1d` | silu | `[['B', 'T', 'n_h_kda*d_head_kda']]` | `None` | `[['B', 'T', 'n_h_kda*d_head_kda']]` |
| prefill | `model.layers.*.self_attn.f_a_proj` | matmul | `[['B*T', 'd_model'], ['d_model', 'd_head_kda']]` | `['d_head_kda', 'd_model']` | `[['B*T', 'd_head_kda']]` |
| prefill | `model.layers.*.self_attn.f_b_proj` | matmul | `[['B*T', 'd_head_kda'], ['d_head_kda', 'n_h_kda*d_head_kda']]` | `['n_h_kda*d_head_kda', 'd_head_kda']` | `[['B*T', 'n_h_kda*d_head_kda']]` |
| prefill | `model.layers.*.self_attn.b_proj` | matmul | `[['B*T', 'd_model'], ['d_model', 'n_h_kda']]` | `['n_h_kda', 'd_model']` | `[['B*T', 'n_h_kda']]` |
| prefill | `model.layers.*.self_attn` | exp | `[['n_h_kda', '1']]` | `None` | `[['n_h_kda', '1']]` |
| prefill | `model.layers.*.self_attn` | sigmoid | `[['B', 'T', 'n_h_kda', 'd_head_kda']]` | `None` | `[['B', 'T', 'n_h_kda', 'd_head_kda']]` |
| prefill | `model.layers.*.self_attn` | sigmoid | `[['B', 'T', 'n_h_kda']]` | `None` | `[['B', 'T', 'n_h_kda']]` |
| prefill | `model.layers.*.self_attn` | exp | `[['B', 'n_h_kda', '5', 'd_chunk', 'd_head_kda']]` | `None` | `[['B', 'n_h_kda', '5', 'd_chunk', 'd_head_kda']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['B*n_h_kda*n_chunk', 'd_chunk', 'd_head_kda'], ['B*n_h_kda*n_chunk', 'd_head_kda', '1']]` | `None` | `[['B*n_h_kda*n_chunk', 'd_chunk', '1']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['B*n_h_kda*n_chunk', 'd_chunk', 'd_chunk'], ['B*n_h_kda*n_chunk', 'd_chunk', 'd_head_kda']]` | `None` | `[['B*n_h_kda*n_chunk', 'd_chunk', 'd_head_kda']]` |
| prefill | `model.layers.*.self_attn` | exp | `[['B', 'n_h_kda', 'd_chunk', 'd_head_kda']]` | `None` | `[['B', 'n_h_kda', 'd_chunk', 'd_head_kda']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['B*n_h_kda', 'd_chunk', 'd_head_kda'], ['B*n_h_kda', 'd_head_kda', '1']]` | `None` | `[['B*n_h_kda', 'd_chunk', '1']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['B*n_h_kda', 'd_chunk', 'd_head_kda'], ['B*n_h_kda', 'd_head_kda', 'd_head_kda']]` | `None` | `[['B*n_h_kda', 'd_chunk', 'd_head_kda']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['B*n_h_kda', 'd_chunk', 'd_chunk'], ['B*n_h_kda', 'd_chunk', 'd_head_kda']]` | `None` | `[['B*n_h_kda', 'd_chunk', 'd_head_kda']]` |
| prefill | `model.layers.*.self_attn` | exp | `[['B', 'n_h_kda', 'd_head_kda']]` | `None` | `[['B', 'n_h_kda', 'd_head_kda']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['B*n_h_kda', 'd_head_kda', 'd_chunk'], ['B*n_h_kda', 'd_chunk', 'd_head_kda']]` | `None` | `[['B*n_h_kda', 'd_head_kda', 'd_head_kda']]` |
| prefill | `model.layers.*.self_attn.g_proj` | matmul | `[['B*T', 'd_model'], ['d_model', 'n_h_kda*d_head_kda']]` | `['n_h_kda*d_head_kda', 'd_model']` | `[['B*T', 'n_h_kda*d_head_kda']]` |
| prefill | `model.layers.*.self_attn.o_norm` | rmsnorm | `[['B', 'T', 'n_h_kda', 'd_head_kda']]` | `['d_head_kda']` | `[['B', 'T', 'n_h_kda', 'd_head_kda']]` |
| prefill | `model.layers.*.self_attn.o_proj` | matmul | `[['B*T', 'n_h_kda*d_head_kda'], ['n_h_kda*d_head_kda', 'd_model']]` | `['d_model', 'n_h_kda*d_head_kda']` | `[['B*T', 'd_model']]` |
| prefill | `model.layers.*` | concat | `[['B*T', '1', 'd_model'], ['B*T', '1', 'd_model']]` | `None` | `[['B*T', '2', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_mul | `[['B*T', '2', 'd_model'], ['B*T', '2', '1']]` | `None` | `[['B*T', '2', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_mul | `[['d_model'], ['d_model']]` | `None` | `[['d_model']]` |
| prefill | `model.layers.*` | elementwise_mul | `[['B*T', '2', 'd_model'], ['d_model']]` | `None` | `[['B*T', '2', 'd_model']]` |
| prefill | `model.layers.*` | sum | `[['B*T', '2', 'd_model']]` | `None` | `[['B*T', '2']]` |
| prefill | `model.layers.*` | softmax | `[['B*T', '2']]` | `None` | `[['B*T', '2']]` |
| prefill | `model.layers.*` | batched_matmul | `[['B*T', '1', '2'], ['B*T', '2', 'd_model']]` | `None` | `[['B*T', '1', 'd_model']]` |
| prefill | `model.layers.*.post_attention_layernorm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.mlp.gate_proj` | matmul | `[['B*T', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['B*T', 'd_ff']]` |
| prefill | `model.layers.*.mlp.up_proj` | matmul | `[['B*T', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['B*T', 'd_ff']]` |
| prefill | `model.layers.*.mlp` | concat | `[['B', 'T', 'd_ff'], ['B', 'T', 'd_ff']]` | `None` | `[['B', 'T', '2*d_ff']]` |
| prefill | `model.layers.*.mlp.act_fn` | tanh | `[['B', 'T', 'd_ff']]` | `None` | `[['B', 'T', 'd_ff']]` |
| prefill | `model.layers.*.mlp.act_fn` | elementwise_mul | `[['B', 'T', 'd_ff']]` | `None` | `[['B', 'T', 'd_ff']]` |
| prefill | `model.layers.*.mlp.act_fn` | sigmoid | `[['B', 'T', 'd_ff']]` | `None` | `[['B', 'T', 'd_ff']]` |
| prefill | `model.layers.*.mlp.act_fn` | elementwise_mul | `[['B', 'T', 'd_ff'], ['B', 'T', 'd_ff']]` | `None` | `[['B', 'T', 'd_ff']]` |
| prefill | `model.layers.*.mlp.down_proj` | matmul | `[['B*T', 'd_ff'], ['d_ff', 'd_model']]` | `['d_model', 'd_ff']` | `[['B*T', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_add | `[['B', 'T', 'd_model'], ['B', 'T', 'd_model']]` | `None` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.block_sparse_moe.gate` | matmul | `[['B*T', 'd_model'], ['d_model', 'E']]` | `['E', 'd_model']` | `[['B*T', 'E']]` |
| prefill | `model.layers.*.block_sparse_moe.gate` | sigmoid | `[['B*T', 'E']]` | `None` | `[['B*T', 'E']]` |
| prefill | `model.layers.*.block_sparse_moe.routed_expert_down_proj` | matmul | `[['B*T', 'd_model'], ['d_model', 'd_moe_lat']]` | `['d_moe_lat', 'd_model']` | `[['B*T', 'd_moe_lat']]` |
| prefill | `model.layers.*.block_sparse_moe.experts.*.w1` | matmul | `[['3840', 'd_moe_lat'], ['d_moe_lat', 'd_moe']]` | `['d_moe', 'd_moe_lat']` | `[['3840', 'd_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.experts.*.w3` | matmul | `[['3840', 'd_moe_lat'], ['d_moe_lat', 'd_moe']]` | `['d_moe', 'd_moe_lat']` | `[['3840', 'd_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.experts.*` | concat | `[['3840', 'd_moe'], ['3840', 'd_moe']]` | `None` | `[['3840', '2*d_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.experts.*.act_fn` | tanh | `[['3840', 'd_moe']]` | `None` | `[['3840', 'd_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.experts.*.act_fn` | elementwise_mul | `[['3840', 'd_moe']]` | `None` | `[['3840', 'd_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.experts.*.act_fn` | sigmoid | `[['3840', 'd_moe']]` | `None` | `[['3840', 'd_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.experts.*.act_fn` | elementwise_mul | `[['3840', 'd_moe'], ['3840', 'd_moe']]` | `None` | `[['3840', 'd_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.experts.*.w2` | matmul | `[['3840', 'd_moe'], ['d_moe', 'd_moe_lat']]` | `['d_moe_lat', 'd_moe']` | `[['3840', 'd_moe_lat']]` |
| prefill | `model.layers.*.block_sparse_moe` | concat | `[['3840', 'd_moe_lat'], ['3840', 'd_moe_lat'], ['3840', 'd_moe_lat'], ['3840', 'd_moe_lat']]` | `None` | `[['B*k*T', 'd_moe_lat']]` |
| prefill | `model.layers.*.block_sparse_moe` | mul_ | `[['B*T', 'k', 'd_moe_lat'], ['B*T', 'k', '1']]` | `None` | `[['B*T', 'k', 'd_moe_lat']]` |
| prefill | `model.layers.*.block_sparse_moe` | sum | `[['B*T', 'k', 'd_moe_lat']]` | `None` | `[['B*T', 'd_moe_lat']]` |
| prefill | `model.layers.*.block_sparse_moe.routed_expert_norm` | rmsnorm | `[['B*T', 'd_moe_lat']]` | `['d_moe_lat']` | `[['B*T', 'd_moe_lat']]` |
| prefill | `model.layers.*.block_sparse_moe.routed_expert_up_proj` | matmul | `[['B*T', 'd_moe_lat'], ['d_moe_lat', 'd_model']]` | `['d_model', 'd_moe_lat']` | `[['B*T', 'd_model']]` |
| prefill | `model.layers.*.block_sparse_moe.shared_experts.gate_proj` | matmul | `[['B*T', 'd_model'], ['d_model', 'E_shared*d_moe']]` | `['E_shared*d_moe', 'd_model']` | `[['B*T', 'E_shared*d_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.shared_experts.up_proj` | matmul | `[['B*T', 'd_model'], ['d_model', 'E_shared*d_moe']]` | `['E_shared*d_moe', 'd_model']` | `[['B*T', 'E_shared*d_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.shared_experts` | concat | `[['B', 'T', 'E_shared*d_moe'], ['B', 'T', 'E_shared*d_moe']]` | `None` | `[['B', 'T', '2*E_shared*d_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.shared_experts.act_fn` | tanh | `[['B', 'T', 'E_shared*d_moe']]` | `None` | `[['B', 'T', 'E_shared*d_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.shared_experts.act_fn` | elementwise_mul | `[['B', 'T', 'E_shared*d_moe']]` | `None` | `[['B', 'T', 'E_shared*d_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.shared_experts.act_fn` | sigmoid | `[['B', 'T', 'E_shared*d_moe']]` | `None` | `[['B', 'T', 'E_shared*d_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.shared_experts.act_fn` | elementwise_mul | `[['B', 'T', 'E_shared*d_moe'], ['B', 'T', 'E_shared*d_moe']]` | `None` | `[['B', 'T', 'E_shared*d_moe']]` |
| prefill | `model.layers.*.block_sparse_moe.shared_experts.down_proj` | matmul | `[['B*T', 'E_shared*d_moe'], ['E_shared*d_moe', 'd_model']]` | `['d_model', 'E_shared*d_moe']` | `[['B*T', 'd_model']]` |
| prefill | `model.layers.*.block_sparse_moe` | elementwise_add | `[['B', 'T', 'd_model'], ['B', 'T', 'd_model']]` | `None` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.self_attn.q_a_proj` | matmul | `[['B*T', 'd_model'], ['d_model', 'c_q']]` | `['c_q', 'd_model']` | `[['B*T', 'c_q']]` |
| prefill | `model.layers.*.self_attn.q_a_layernorm` | rmsnorm | `[['B', 'T', 'c_q']]` | `['c_q']` | `[['B', 'T', 'c_q']]` |
| prefill | `model.layers.*.self_attn.q_b_proj` | matmul | `[['B*T', 'c_q'], ['c_q', 'n_h*(d_nope+d_rope)']]` | `['n_h*(d_nope+d_rope)', 'c_q']` | `[['B*T', 'n_h*(d_nope+d_rope)']]` |
| prefill | `model.layers.*.self_attn.kv_a_proj_with_mqa` | matmul | `[['B*T', 'd_model'], ['d_model', 'c_kv+d_rope']]` | `['c_kv+d_rope', 'd_model']` | `[['B*T', 'c_kv+d_rope']]` |
| prefill | `model.layers.*.self_attn.kv_a_layernorm` | rmsnorm | `[['B', 'T', 'c_kv']]` | `['c_kv']` | `[['B', 'T', 'c_kv']]` |
| prefill | `model.layers.*.self_attn.kv_b_proj` | matmul | `[['B*T', 'c_kv'], ['c_kv', 'n_h*(d_nope+d_v)']]` | `['n_h*(d_nope+d_v)', 'c_kv']` | `[['B*T', 'n_h*(d_nope+d_v)']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['B*n_h', 'T', 'd_nope+d_rope'], ['B*n_h', 'd_nope+d_rope', 'T']]` | `None` | `[['B*n_h', 'T', 'T']]` |
| prefill | `model.layers.*.self_attn` | softmax | `[['B', 'n_h', 'T', 'T']]` | `None` | `[['B', 'n_h', 'T', 'T']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['B*n_h', 'T', 'T'], ['B*n_h', 'T', 'd_nope']]` | `None` | `[['B*n_h', 'T', 'd_nope']]` |
| prefill | `model.layers.*.self_attn.g_proj` | matmul | `[['B*T', 'd_model'], ['d_model', 'n_h*d_v']]` | `['n_h*d_v', 'd_model']` | `[['B*T', 'n_h*d_v']]` |
| prefill | `model.layers.*.self_attn` | sigmoid | `[['B', 'T', 'n_h*d_v']]` | `None` | `[['B', 'T', 'n_h*d_v']]` |
| prefill | `model.layers.*.self_attn.o_proj` | matmul | `[['B*T', 'n_h*d_v'], ['n_h*d_v', 'd_model']]` | `['d_model', 'n_h*d_v']` | `[['B*T', 'd_model']]` |
| prefill | `model.layers.*` | concat | `[['B*T', '2', 'd_model'], ['B*T', '1', 'd_model']]` | `None` | `[['B*T', '3', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_mul | `[['B*T', '3', 'd_model'], ['B*T', '3', '1']]` | `None` | `[['B*T', '3', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_mul | `[['B*T', '3', 'd_model'], ['d_model']]` | `None` | `[['B*T', '3', 'd_model']]` |
| prefill | `model.layers.*` | sum | `[['B*T', '3', 'd_model']]` | `None` | `[['B*T', '3']]` |
| prefill | `model.layers.*` | softmax | `[['B*T', '3']]` | `None` | `[['B*T', '3']]` |
| prefill | `model.layers.*` | batched_matmul | `[['B*T', '1', '3'], ['B*T', '3', 'd_model']]` | `None` | `[['B*T', '1', 'd_model']]` |
| prefill | `model.layers.*` | concat | `[['B*T', '3', 'd_model'], ['B*T', '1', 'd_model']]` | `None` | `[['B*T', '4', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_mul | `[['B*T', '4', 'd_model'], ['B*T', '4', '1']]` | `None` | `[['B*T', '4', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_mul | `[['B*T', '4', 'd_model'], ['d_model']]` | `None` | `[['B*T', '4', 'd_model']]` |
| prefill | `model.layers.*` | sum | `[['B*T', '4', 'd_model']]` | `None` | `[['B*T', '4']]` |
| prefill | `model.layers.*` | softmax | `[['B*T', '4']]` | `None` | `[['B*T', '4']]` |
| prefill | `model.layers.*` | batched_matmul | `[['B*T', '1', '4'], ['B*T', '4', 'd_model']]` | `None` | `[['B*T', '1', 'd_model']]` |
| prefill | `model.layers.*` | concat | `[['B*T', '4', 'd_model'], ['B*T', '1', 'd_model']]` | `None` | `[['B*T', '5', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_mul | `[['B*T', '5', 'd_model'], ['B*T', '5', '1']]` | `None` | `[['B*T', '5', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_mul | `[['B*T', '5', 'd_model'], ['d_model']]` | `None` | `[['B*T', '5', 'd_model']]` |
| prefill | `model.layers.*` | sum | `[['B*T', '5', 'd_model']]` | `None` | `[['B*T', '5']]` |
| prefill | `model.layers.*` | softmax | `[['B*T', '5']]` | `None` | `[['B*T', '5']]` |
| prefill | `model.layers.*` | batched_matmul | `[['B*T', '1', '5'], ['B*T', '5', 'd_model']]` | `None` | `[['B*T', '1', 'd_model']]` |
| prefill | `model.layers.*` | concat | `[['B*T', '5', 'd_model'], ['B*T', '1', 'd_model']]` | `None` | `[['B*T', '6', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_mul | `[['B*T', '6', 'd_model'], ['B*T', '6', '1']]` | `None` | `[['B*T', '6', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_mul | `[['B*T', '6', 'd_model'], ['d_model']]` | `None` | `[['B*T', '6', 'd_model']]` |
| prefill | `model.layers.*` | sum | `[['B*T', '6', 'd_model']]` | `None` | `[['B*T', '6']]` |
| prefill | `model.layers.*` | softmax | `[['B*T', '6']]` | `None` | `[['B*T', '6']]` |
| prefill | `model.layers.*` | batched_matmul | `[['B*T', '1', '6'], ['B*T', '6', 'd_model']]` | `None` | `[['B*T', '1', 'd_model']]` |
| prefill | `model.layers.*` | concat | `[['B*T', '6', 'd_model'], ['B*T', '1', 'd_model']]` | `None` | `[['B*T', '7', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_mul | `[['B*T', '7', 'd_model'], ['B*T', '7', '1']]` | `None` | `[['B*T', '7', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_mul | `[['B*T', '7', 'd_model'], ['d_model']]` | `None` | `[['B*T', '7', 'd_model']]` |
| prefill | `model.layers.*` | sum | `[['B*T', '7', 'd_model']]` | `None` | `[['B*T', '7']]` |
| prefill | `model.layers.*` | softmax | `[['B*T', '7']]` | `None` | `[['B*T', '7']]` |
| prefill | `model.layers.*` | batched_matmul | `[['B*T', '1', '7'], ['B*T', '7', 'd_model']]` | `None` | `[['B*T', '1', 'd_model']]` |
| prefill | `model.layers.*` | concat | `[['B*T', '7', 'd_model'], ['B*T', '1', 'd_model']]` | `None` | `[['B*T', '8', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_mul | `[['B*T', '8', 'd_model'], ['B*T', '8', '1']]` | `None` | `[['B*T', '8', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_mul | `[['B*T', '8', 'd_model'], ['d_model']]` | `None` | `[['B*T', '8', 'd_model']]` |
| prefill | `model.layers.*` | sum | `[['B*T', '8', 'd_model']]` | `None` | `[['B*T', '8']]` |
| prefill | `model.layers.*` | softmax | `[['B*T', '8']]` | `None` | `[['B*T', '8']]` |
| prefill | `model.layers.*` | batched_matmul | `[['B*T', '1', '8'], ['B*T', '8', 'd_model']]` | `None` | `[['B*T', '1', 'd_model']]` |
| prefill | `model.layers.*` | concat | `[['B*T', '8', 'd_model'], ['B*T', '1', 'd_model']]` | `None` | `[['B*T', '9', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_mul | `[['B*T', '9', 'd_model'], ['B*T', '9', '1']]` | `None` | `[['B*T', '9', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_mul | `[['B*T', '9', 'd_model'], ['d_model']]` | `None` | `[['B*T', '9', 'd_model']]` |
| prefill | `model.layers.*` | sum | `[['B*T', '9', 'd_model']]` | `None` | `[['B*T', '9']]` |
| prefill | `model.layers.*` | softmax | `[['B*T', '9']]` | `None` | `[['B*T', '9']]` |
| prefill | `model.layers.*` | batched_matmul | `[['B*T', '1', '9'], ['B*T', '9', 'd_model']]` | `None` | `[['B*T', '1', 'd_model']]` |
| prefill | `model` | concat | `[['B*T', '8', 'd_model'], ['B*T', '1', 'd_model']]` | `None` | `[['B*T', '9', 'd_model']]` |
| prefill | `model` | elementwise_mul | `[['B*T', '9', 'd_model'], ['B*T', '9', '1']]` | `None` | `[['B*T', '9', 'd_model']]` |
| prefill | `model` | elementwise_mul | `[['d_model'], ['d_model']]` | `None` | `[['d_model']]` |
| prefill | `model` | elementwise_mul | `[['B*T', '9', 'd_model'], ['d_model']]` | `None` | `[['B*T', '9', 'd_model']]` |
| prefill | `model` | sum | `[['B*T', '9', 'd_model']]` | `None` | `[['B*T', '9']]` |
| prefill | `model` | softmax | `[['B*T', '9']]` | `None` | `[['B*T', '9']]` |
| prefill | `model` | batched_matmul | `[['B*T', '1', '9'], ['B*T', '9', 'd_model']]` | `None` | `[['B*T', '1', 'd_model']]` |
| prefill | `model.norm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `lm_head` | matmul | `[['B*T', 'd_model'], ['d_model', 'V']]` | `['V', 'd_model']` | `[['B*T', 'V']]` |
| decode | `model.embed_tokens` | embedding | `[['V', 'd_model'], ['B', '1']]` | `['V', 'd_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*` | concat | `[['B', '0', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.input_layernorm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.self_attn.q_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_h_kda*d_head_kda']]` | `['n_h_kda*d_head_kda', 'd_model']` | `[['B', 'n_h_kda*d_head_kda']]` |
| decode | `model.layers.*.self_attn.k_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_h_kda*d_head_kda']]` | `['n_h_kda*d_head_kda', 'd_model']` | `[['B', 'n_h_kda*d_head_kda']]` |
| decode | `model.layers.*.self_attn.v_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_h_kda*d_head_kda']]` | `['n_h_kda*d_head_kda', 'd_model']` | `[['B', 'n_h_kda*d_head_kda']]` |
| decode | `model.layers.*.self_attn.q_conv1d` | conv1d | `[['B', 'n_h_kda*d_head_kda', 'd_conv'], ['n_h_kda*d_head_kda', '1', 'd_conv']]` | `['n_h_kda*d_head_kda', '1', 'd_conv']` | `[['B', 'n_h_kda*d_head_kda', '1']]` |
| decode | `model.layers.*.self_attn.q_conv1d` | silu | `[['B', '1', 'n_h_kda*d_head_kda']]` | `None` | `[['B', '1', 'n_h_kda*d_head_kda']]` |
| decode | `model.layers.*.self_attn.k_conv1d` | conv1d | `[['B', 'n_h_kda*d_head_kda', 'd_conv'], ['n_h_kda*d_head_kda', '1', 'd_conv']]` | `['n_h_kda*d_head_kda', '1', 'd_conv']` | `[['B', 'n_h_kda*d_head_kda', '1']]` |
| decode | `model.layers.*.self_attn.k_conv1d` | silu | `[['B', '1', 'n_h_kda*d_head_kda']]` | `None` | `[['B', '1', 'n_h_kda*d_head_kda']]` |
| decode | `model.layers.*.self_attn.v_conv1d` | conv1d | `[['B', 'n_h_kda*d_head_kda', 'd_conv'], ['n_h_kda*d_head_kda', '1', 'd_conv']]` | `['n_h_kda*d_head_kda', '1', 'd_conv']` | `[['B', 'n_h_kda*d_head_kda', '1']]` |
| decode | `model.layers.*.self_attn.v_conv1d` | silu | `[['B', '1', 'n_h_kda*d_head_kda']]` | `None` | `[['B', '1', 'n_h_kda*d_head_kda']]` |
| decode | `model.layers.*.self_attn.f_a_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_head_kda']]` | `['d_head_kda', 'd_model']` | `[['B', 'd_head_kda']]` |
| decode | `model.layers.*.self_attn.f_b_proj` | matmul | `[['B', 'd_head_kda'], ['d_head_kda', 'n_h_kda*d_head_kda']]` | `['n_h_kda*d_head_kda', 'd_head_kda']` | `[['B', 'n_h_kda*d_head_kda']]` |
| decode | `model.layers.*.self_attn.b_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_h_kda']]` | `['n_h_kda', 'd_model']` | `[['B', 'n_h_kda']]` |
| decode | `model.layers.*.self_attn` | exp | `[['n_h_kda', '1']]` | `None` | `[['n_h_kda', '1']]` |
| decode | `model.layers.*.self_attn` | sigmoid | `[['B', '1', 'n_h_kda', 'd_head_kda']]` | `None` | `[['B', '1', 'n_h_kda', 'd_head_kda']]` |
| decode | `model.layers.*.self_attn` | sigmoid | `[['B', '1', 'n_h_kda']]` | `None` | `[['B', '1', 'n_h_kda']]` |
| decode | `model.layers.*.self_attn` | exp | `[['B', 'n_h_kda', 'd_head_kda', '1']]` | `None` | `[['B', 'n_h_kda', 'd_head_kda', '1']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['B*n_h_kda', '1', 'd_head_kda'], ['B*n_h_kda', 'd_head_kda', 'd_head_kda']]` | `None` | `[['B*n_h_kda', '1', 'd_head_kda']]` |
| decode | `model.layers.*.self_attn.g_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_h_kda*d_head_kda']]` | `['n_h_kda*d_head_kda', 'd_model']` | `[['B', 'n_h_kda*d_head_kda']]` |
| decode | `model.layers.*.self_attn.o_norm` | rmsnorm | `[['B', '1', 'n_h_kda', 'd_head_kda']]` | `['d_head_kda']` | `[['B', '1', 'n_h_kda', 'd_head_kda']]` |
| decode | `model.layers.*.self_attn.o_proj` | matmul | `[['B', 'n_h_kda*d_head_kda'], ['n_h_kda*d_head_kda', 'd_model']]` | `['d_model', 'n_h_kda*d_head_kda']` | `[['B', 'd_model']]` |
| decode | `model.layers.*` | concat | `[['B', '1', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '2', 'd_model']]` |
| decode | `model.layers.*` | elementwise_mul | `[['B', '2', 'd_model'], ['B', '2', '1']]` | `None` | `[['B', '2', 'd_model']]` |
| decode | `model.layers.*` | elementwise_mul | `[['d_model'], ['d_model']]` | `None` | `[['d_model']]` |
| decode | `model.layers.*` | elementwise_mul | `[['B', '2', 'd_model'], ['d_model']]` | `None` | `[['B', '2', 'd_model']]` |
| decode | `model.layers.*` | sum | `[['B', '2', 'd_model']]` | `None` | `[['B', '2']]` |
| decode | `model.layers.*` | softmax | `[['B', '2']]` | `None` | `[['B', '2']]` |
| decode | `model.layers.*` | batched_matmul | `[['B', '1', '2'], ['B', '2', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.post_attention_layernorm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.mlp.gate_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['B', 'd_ff']]` |
| decode | `model.layers.*.mlp.up_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['B', 'd_ff']]` |
| decode | `model.layers.*.mlp` | concat | `[['B', '1', 'd_ff'], ['B', '1', 'd_ff']]` | `None` | `[['B', '1', '2*d_ff']]` |
| decode | `model.layers.*.mlp.act_fn` | tanh | `[['B', '1', 'd_ff']]` | `None` | `[['B', '1', 'd_ff']]` |
| decode | `model.layers.*.mlp.act_fn` | elementwise_mul | `[['B', '1', 'd_ff']]` | `None` | `[['B', '1', 'd_ff']]` |
| decode | `model.layers.*.mlp.act_fn` | sigmoid | `[['B', '1', 'd_ff']]` | `None` | `[['B', '1', 'd_ff']]` |
| decode | `model.layers.*.mlp.act_fn` | elementwise_mul | `[['B', '1', 'd_ff'], ['B', '1', 'd_ff']]` | `None` | `[['B', '1', 'd_ff']]` |
| decode | `model.layers.*.mlp.down_proj` | matmul | `[['B', 'd_ff'], ['d_ff', 'd_model']]` | `['d_model', 'd_ff']` | `[['B', 'd_model']]` |
| decode | `model.layers.*` | elementwise_add | `[['B', '1', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.block_sparse_moe.gate` | matmul | `[['B', 'd_model'], ['d_model', 'E']]` | `['E', 'd_model']` | `[['B', 'E']]` |
| decode | `model.layers.*.block_sparse_moe.gate` | sigmoid | `[['B', 'E']]` | `None` | `[['B', 'E']]` |
| decode | `model.layers.*.block_sparse_moe.routed_expert_down_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_moe_lat']]` | `['d_moe_lat', 'd_model']` | `[['B', 'd_moe_lat']]` |
| decode | `model.layers.*.block_sparse_moe.experts.*.w1` | matmul | `[['12', 'd_moe_lat'], ['d_moe_lat', 'd_moe']]` | `['d_moe', 'd_moe_lat']` | `[['12', 'd_moe']]` |
| decode | `model.layers.*.block_sparse_moe.experts.*.w3` | matmul | `[['12', 'd_moe_lat'], ['d_moe_lat', 'd_moe']]` | `['d_moe', 'd_moe_lat']` | `[['12', 'd_moe']]` |
| decode | `model.layers.*.block_sparse_moe.experts.*` | concat | `[['12', 'd_moe'], ['12', 'd_moe']]` | `None` | `[['12', '2*d_moe']]` |
| decode | `model.layers.*.block_sparse_moe.experts.*.act_fn` | tanh | `[['12', 'd_moe']]` | `None` | `[['12', 'd_moe']]` |
| decode | `model.layers.*.block_sparse_moe.experts.*.act_fn` | elementwise_mul | `[['12', 'd_moe']]` | `None` | `[['12', 'd_moe']]` |
| decode | `model.layers.*.block_sparse_moe.experts.*.act_fn` | sigmoid | `[['12', 'd_moe']]` | `None` | `[['12', 'd_moe']]` |
| decode | `model.layers.*.block_sparse_moe.experts.*.act_fn` | elementwise_mul | `[['12', 'd_moe'], ['12', 'd_moe']]` | `None` | `[['12', 'd_moe']]` |
| decode | `model.layers.*.block_sparse_moe.experts.*.w2` | matmul | `[['12', 'd_moe'], ['d_moe', 'd_moe_lat']]` | `['d_moe_lat', 'd_moe']` | `[['12', 'd_moe_lat']]` |
| decode | `model.layers.*.block_sparse_moe` | concat | `[['12', 'd_moe_lat'], ['12', 'd_moe_lat'], ['12', 'd_moe_lat'], ['12', 'd_moe_lat']]` | `None` | `[['B*k', 'd_moe_lat']]` |
| decode | `model.layers.*.block_sparse_moe` | mul_ | `[['B', 'k', 'd_moe_lat'], ['B', 'k', '1']]` | `None` | `[['B', 'k', 'd_moe_lat']]` |
| decode | `model.layers.*.block_sparse_moe` | sum | `[['B', 'k', 'd_moe_lat']]` | `None` | `[['B', 'd_moe_lat']]` |
| decode | `model.layers.*.block_sparse_moe.routed_expert_norm` | rmsnorm | `[['B', 'd_moe_lat']]` | `['d_moe_lat']` | `[['B', 'd_moe_lat']]` |
| decode | `model.layers.*.block_sparse_moe.routed_expert_up_proj` | matmul | `[['B', 'd_moe_lat'], ['d_moe_lat', 'd_model']]` | `['d_model', 'd_moe_lat']` | `[['B', 'd_model']]` |
| decode | `model.layers.*.block_sparse_moe.shared_experts.gate_proj` | matmul | `[['B', 'd_model'], ['d_model', 'E_shared*d_moe']]` | `['E_shared*d_moe', 'd_model']` | `[['B', 'E_shared*d_moe']]` |
| decode | `model.layers.*.block_sparse_moe.shared_experts.up_proj` | matmul | `[['B', 'd_model'], ['d_model', 'E_shared*d_moe']]` | `['E_shared*d_moe', 'd_model']` | `[['B', 'E_shared*d_moe']]` |
| decode | `model.layers.*.block_sparse_moe.shared_experts` | concat | `[['B', '1', 'E_shared*d_moe'], ['B', '1', 'E_shared*d_moe']]` | `None` | `[['B', '1', '2*E_shared*d_moe']]` |
| decode | `model.layers.*.block_sparse_moe.shared_experts.act_fn` | tanh | `[['B', '1', 'E_shared*d_moe']]` | `None` | `[['B', '1', 'E_shared*d_moe']]` |
| decode | `model.layers.*.block_sparse_moe.shared_experts.act_fn` | elementwise_mul | `[['B', '1', 'E_shared*d_moe']]` | `None` | `[['B', '1', 'E_shared*d_moe']]` |
| decode | `model.layers.*.block_sparse_moe.shared_experts.act_fn` | sigmoid | `[['B', '1', 'E_shared*d_moe']]` | `None` | `[['B', '1', 'E_shared*d_moe']]` |
| decode | `model.layers.*.block_sparse_moe.shared_experts.act_fn` | elementwise_mul | `[['B', '1', 'E_shared*d_moe'], ['B', '1', 'E_shared*d_moe']]` | `None` | `[['B', '1', 'E_shared*d_moe']]` |
| decode | `model.layers.*.block_sparse_moe.shared_experts.down_proj` | matmul | `[['B', 'E_shared*d_moe'], ['E_shared*d_moe', 'd_model']]` | `['d_model', 'E_shared*d_moe']` | `[['B', 'd_model']]` |
| decode | `model.layers.*.block_sparse_moe` | elementwise_add | `[['B', '1', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.self_attn.q_a_proj` | matmul | `[['B', 'd_model'], ['d_model', 'c_q']]` | `['c_q', 'd_model']` | `[['B', 'c_q']]` |
| decode | `model.layers.*.self_attn.q_a_layernorm` | rmsnorm | `[['B', '1', 'c_q']]` | `['c_q']` | `[['B', '1', 'c_q']]` |
| decode | `model.layers.*.self_attn.q_b_proj` | matmul | `[['B', 'c_q'], ['c_q', 'n_h*(d_nope+d_rope)']]` | `['n_h*(d_nope+d_rope)', 'c_q']` | `[['B', 'n_h*(d_nope+d_rope)']]` |
| decode | `model.layers.*.self_attn.kv_a_proj_with_mqa` | matmul | `[['B', 'd_model'], ['d_model', 'c_kv+d_rope']]` | `['c_kv+d_rope', 'd_model']` | `[['B', 'c_kv+d_rope']]` |
| decode | `model.layers.*.self_attn.kv_a_layernorm` | rmsnorm | `[['B', '1', 'c_kv']]` | `['c_kv']` | `[['B', '1', 'c_kv']]` |
| decode | `model.layers.*.self_attn.kv_b_proj` | matmul | `[['B', 'c_kv'], ['c_kv', 'n_h*(d_nope+d_v)']]` | `['n_h*(d_nope+d_v)', 'c_kv']` | `[['B', 'n_h*(d_nope+d_v)']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['B*n_h', '1', 'd_nope+d_rope'], ['B*n_h', 'd_nope+d_rope', 'T+1']]` | `None` | `[['B*n_h', '1', 'T+1']]` |
| decode | `model.layers.*.self_attn` | softmax | `[['B', 'n_h', '1', 'T+1']]` | `None` | `[['B', 'n_h', '1', 'T+1']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['B*n_h', '1', 'T+1'], ['B*n_h', 'T+1', 'd_nope']]` | `None` | `[['B*n_h', '1', 'd_nope']]` |
| decode | `model.layers.*.self_attn.g_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_h*d_v']]` | `['n_h*d_v', 'd_model']` | `[['B', 'n_h*d_v']]` |
| decode | `model.layers.*.self_attn` | sigmoid | `[['B', '1', 'n_h*d_v']]` | `None` | `[['B', '1', 'n_h*d_v']]` |
| decode | `model.layers.*.self_attn.o_proj` | matmul | `[['B', 'n_h*d_v'], ['n_h*d_v', 'd_model']]` | `['d_model', 'n_h*d_v']` | `[['B', 'd_model']]` |
| decode | `model.layers.*` | concat | `[['B', '2', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '3', 'd_model']]` |
| decode | `model.layers.*` | elementwise_mul | `[['B', '3', 'd_model'], ['B', '3', '1']]` | `None` | `[['B', '3', 'd_model']]` |
| decode | `model.layers.*` | elementwise_mul | `[['B', '3', 'd_model'], ['d_model']]` | `None` | `[['B', '3', 'd_model']]` |
| decode | `model.layers.*` | sum | `[['B', '3', 'd_model']]` | `None` | `[['B', '3']]` |
| decode | `model.layers.*` | softmax | `[['B', '3']]` | `None` | `[['B', '3']]` |
| decode | `model.layers.*` | batched_matmul | `[['B', '1', '3'], ['B', '3', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*` | concat | `[['B', '3', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '4', 'd_model']]` |
| decode | `model.layers.*` | elementwise_mul | `[['B', '4', 'd_model'], ['B', '4', '1']]` | `None` | `[['B', '4', 'd_model']]` |
| decode | `model.layers.*` | elementwise_mul | `[['B', '4', 'd_model'], ['d_model']]` | `None` | `[['B', '4', 'd_model']]` |
| decode | `model.layers.*` | sum | `[['B', '4', 'd_model']]` | `None` | `[['B', '4']]` |
| decode | `model.layers.*` | softmax | `[['B', '4']]` | `None` | `[['B', '4']]` |
| decode | `model.layers.*` | batched_matmul | `[['B', '1', '4'], ['B', '4', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*` | concat | `[['B', '4', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '5', 'd_model']]` |
| decode | `model.layers.*` | elementwise_mul | `[['B', '5', 'd_model'], ['B', '5', '1']]` | `None` | `[['B', '5', 'd_model']]` |
| decode | `model.layers.*` | elementwise_mul | `[['B', '5', 'd_model'], ['d_model']]` | `None` | `[['B', '5', 'd_model']]` |
| decode | `model.layers.*` | sum | `[['B', '5', 'd_model']]` | `None` | `[['B', '5']]` |
| decode | `model.layers.*` | softmax | `[['B', '5']]` | `None` | `[['B', '5']]` |
| decode | `model.layers.*` | batched_matmul | `[['B', '1', '5'], ['B', '5', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*` | concat | `[['B', '5', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '6', 'd_model']]` |
| decode | `model.layers.*` | elementwise_mul | `[['B', '6', 'd_model'], ['B', '6', '1']]` | `None` | `[['B', '6', 'd_model']]` |
| decode | `model.layers.*` | elementwise_mul | `[['B', '6', 'd_model'], ['d_model']]` | `None` | `[['B', '6', 'd_model']]` |
| decode | `model.layers.*` | sum | `[['B', '6', 'd_model']]` | `None` | `[['B', '6']]` |
| decode | `model.layers.*` | softmax | `[['B', '6']]` | `None` | `[['B', '6']]` |
| decode | `model.layers.*` | batched_matmul | `[['B', '1', '6'], ['B', '6', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*` | concat | `[['B', '6', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '7', 'd_model']]` |
| decode | `model.layers.*` | elementwise_mul | `[['B', '7', 'd_model'], ['B', '7', '1']]` | `None` | `[['B', '7', 'd_model']]` |
| decode | `model.layers.*` | elementwise_mul | `[['B', '7', 'd_model'], ['d_model']]` | `None` | `[['B', '7', 'd_model']]` |
| decode | `model.layers.*` | sum | `[['B', '7', 'd_model']]` | `None` | `[['B', '7']]` |
| decode | `model.layers.*` | softmax | `[['B', '7']]` | `None` | `[['B', '7']]` |
| decode | `model.layers.*` | batched_matmul | `[['B', '1', '7'], ['B', '7', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*` | concat | `[['B', '7', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '8', 'd_model']]` |
| decode | `model.layers.*` | elementwise_mul | `[['B', '8', 'd_model'], ['B', '8', '1']]` | `None` | `[['B', '8', 'd_model']]` |
| decode | `model.layers.*` | elementwise_mul | `[['B', '8', 'd_model'], ['d_model']]` | `None` | `[['B', '8', 'd_model']]` |
| decode | `model.layers.*` | sum | `[['B', '8', 'd_model']]` | `None` | `[['B', '8']]` |
| decode | `model.layers.*` | softmax | `[['B', '8']]` | `None` | `[['B', '8']]` |
| decode | `model.layers.*` | batched_matmul | `[['B', '1', '8'], ['B', '8', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*` | concat | `[['B', '8', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '9', 'd_model']]` |
| decode | `model.layers.*` | elementwise_mul | `[['B', '9', 'd_model'], ['B', '9', '1']]` | `None` | `[['B', '9', 'd_model']]` |
| decode | `model.layers.*` | elementwise_mul | `[['B', '9', 'd_model'], ['d_model']]` | `None` | `[['B', '9', 'd_model']]` |
| decode | `model.layers.*` | sum | `[['B', '9', 'd_model']]` | `None` | `[['B', '9']]` |
| decode | `model.layers.*` | softmax | `[['B', '9']]` | `None` | `[['B', '9']]` |
| decode | `model.layers.*` | batched_matmul | `[['B', '1', '9'], ['B', '9', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model` | concat | `[['B', '8', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '9', 'd_model']]` |
| decode | `model` | elementwise_mul | `[['B', '9', 'd_model'], ['B', '9', '1']]` | `None` | `[['B', '9', 'd_model']]` |
| decode | `model` | elementwise_mul | `[['d_model'], ['d_model']]` | `None` | `[['d_model']]` |
| decode | `model` | elementwise_mul | `[['B', '9', 'd_model'], ['d_model']]` | `None` | `[['B', '9', 'd_model']]` |
| decode | `model` | sum | `[['B', '9', 'd_model']]` | `None` | `[['B', '9']]` |
| decode | `model` | softmax | `[['B', '9']]` | `None` | `[['B', '9']]` |
| decode | `model` | batched_matmul | `[['B', '1', '9'], ['B', '9', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.norm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `lm_head` | matmul | `[['B', 'd_model'], ['d_model', 'V']]` | `['V', 'd_model']` | `[['B', 'V']]` |

## 전수 점검 — 이 모델이 쓰는 이름 전부

위 절이 '풀리지 않은 것'이라면 여기는 **전부**다. 규칙이 자신 있게 붙인 이름도 틀릴 수 있고, 그런 건 미결 목록에 절대 오르지 않는다. 한 줄씩 읽고 **그 모듈에서 그 이름이 말이 되는지** 보라.

### A. 붙은 이름 전부 (40종)

| 라벨 | 값 | 나타나는 모듈 | 축 수 |
|---|---|---|---|
| `B` |  | `model.layers.*.self_attn`, `model.layers.*.block_sparse_moe.shared_experts.act_fn`, `model.layers.*.input_layernorm`, `model.layers.*.post_attention_layernorm` 외 132개 | 1224161 |
| `n_h_kda` | 96 | `model.layers.*.self_attn`, `model.layers.*.self_attn.o_norm`, `model.layers.*.self_attn.b_proj` | 1175691 |
| `d_chunk` | 64 | `model.layers.*.self_attn` | 996843 |
| `d_head_kda` | 128 | `model.layers.*.self_attn`, `model.layers.*.self_attn.o_norm`, `model.layers.*.self_attn.f_a_proj`, `model.layers.*.self_attn.f_b_proj` | 891963 |
| `B*n_h_kda` |  | `model.layers.*.self_attn` | 141174 |
| `d_model` | 7168 | `model.layers.*.block_sparse_moe.gate`, `model.layers.*.input_layernorm`, `model.layers.*.post_attention_layernorm`, `model.layers.*.self_attn.g_proj` 외 114개 | 40934 |
| `d_moe` | 3072 | `model.layers.*.block_sparse_moe.experts.*.act_fn`, `model.layers.*.block_sparse_moe.experts.*.w1`, `model.layers.*.block_sparse_moe.experts.*.w3`, `model.layers.*.block_sparse_moe.experts.*.w2` 외 4개 | 35328 |
| `B*n_h_kda*n_chunk` |  | `model.layers.*.self_attn` | 27324 |
| `d_moe_lat` | 3584 | `model.layers.*.block_sparse_moe`, `model.layers.*.block_sparse_moe.experts.*.w1`, `model.layers.*.block_sparse_moe.experts.*.w3`, `model.layers.*.block_sparse_moe.experts.*.w2` 외 3개 | 23184 |
| `T` |  | `model.layers.*.self_attn`, `model.layers.*.block_sparse_moe.shared_experts.act_fn`, `model.layers.*.input_layernorm`, `model.layers.*.post_attention_layernorm` 외 129개 | 20839 |
| `B*T` |  | `model.layers.*.block_sparse_moe.gate`, `model.layers.*.block_sparse_moe`, `model.layers.*.block_sparse_moe.routed_expert_norm`, `model.layers.*.self_attn.g_proj` 외 114개 | 18579 |
| `n_h_kda*d_head_kda` |  | `model.layers.*.self_attn.q_conv1d`, `model.layers.*.self_attn.k_conv1d`, `model.layers.*.self_attn.v_conv1d`, `model.layers.*.self_attn.o_proj` 외 9개 | 14145 |
| `E_shared*d_moe` |  | `model.layers.*.block_sparse_moe.shared_experts.act_fn`, `model.layers.*.block_sparse_moe.shared_experts.gate_proj`, `model.layers.*.block_sparse_moe.shared_experts.up_proj`, `model.layers.*.block_sparse_moe.shared_experts.down_proj` 외 1개 | 9936 |
| `E` | 896 | `model.layers.*.block_sparse_moe.gate`, `model.layers.*.block_sparse_moe` | 4232 |
| `k` | 16 | `model.layers.*.block_sparse_moe`, `model.layers.*.block_sparse_moe.gate` | 3864 |
| `n_h` | 96 | `model.layers.*.self_attn` | 3600 |
| `2*d_moe` |  | `model.layers.*.block_sparse_moe.experts.*.act_fn`, `model.layers.*.block_sparse_moe.experts.0`, `model.layers.*.block_sparse_moe.experts.1`, `model.layers.*.block_sparse_moe.experts.2` 외 1개 | 2208 |
| `B*k*T` |  | `model.layers.*.block_sparse_moe` | 1840 |
| `B*k` |  | `model.layers.*.block_sparse_moe` | 1840 |
| `d_conv` | 4 | `model.layers.*.self_attn.q_conv1d`, `model.layers.*.self_attn.k_conv1d`, `model.layers.*.self_attn.v_conv1d`, `model.layers.*.self_attn.q_conv1d.conv` 외 2개 | 1449 |
| `c_q` | 1536 | `model.layers.*.self_attn.q_a_layernorm`, `model.layers.*.self_attn.q_a_proj`, `model.layers.*.self_attn.q_b_proj` | 1344 |
| `d_nope+d_rope` |  | `model.layers.*.self_attn` | 1224 |
| `T+1` |  | `model.layers.*.self_attn`, `model` | 1191 |
| `n_h*d_v` |  | `model.layers.*.self_attn.g_proj`, `model.layers.*.self_attn.o_proj`, `model.layers.*.self_attn` | 1056 |
| `c_kv` | 512 | `model.layers.*.self_attn.kv_a_layernorm`, `model.layers.*.self_attn.kv_b_proj`, `model.layers.*.self_attn` | 1008 |
| `d_nope` | 128 | `model.layers.*.self_attn` | 936 |
| `d_rope/2` |  | `model.layers.*.self_attn` | 759 |
| `B*n_h` |  | `model.layers.*.self_attn` | 576 |
| `2*E_shared*d_moe` |  | `model.layers.*.block_sparse_moe.shared_experts.act_fn`, `model.layers.*.block_sparse_moe.shared_experts` | 552 |
| `n_h*(d_nope+d_rope)` |  | `model.layers.*.self_attn.q_b_proj`, `model.layers.*.self_attn` | 432 |
| `c_kv+d_rope` |  | `model.layers.*.self_attn.kv_a_proj_with_mqa`, `model.layers.*.self_attn` | 432 |
| `n_h*(d_nope+d_v)` |  | `model.layers.*.self_attn.kv_b_proj`, `model.layers.*.self_attn` | 432 |
| `T+d_conv-1` |  | `model.layers.*.self_attn.q_conv1d.conv`, `model.layers.*.self_attn.q_conv1d`, `model.layers.*.self_attn.k_conv1d.conv`, `model.layers.*.self_attn.k_conv1d` 외 2개 | 414 |
| `d_rope` | 64 | `model.layers.*.self_attn` | 384 |
| `d_v` | 128 | `model.layers.*.self_attn` | 384 |
| `d_nope+d_v` |  | `model.layers.*.self_attn` | 192 |
| `n_chunk` |  | `model.layers.*.self_attn` | 138 |
| `d_ff` | 33792 | `model.layers.*.mlp.act_fn`, `model.layers.*.mlp.gate_proj`, `model.layers.*.mlp.up_proj`, `model.layers.*.mlp.down_proj` 외 1개 | 108 |
| `V` | 163840 | `lm_head`, `model.embed_tokens` | 20 |
| `2*d_ff` |  | `model.layers.*.mlp.act_fn`, `model.layers.*.mlp` | 6 |

### B. 이름 없이 남은 정수 전부 (266쌍)

**여기가 필터가 못 보던 자리다.** 정수가 남는 것 자체는 정상이다(루프 인덱스, 피연산자 개수, 브로드캐스트 축). 문제는 **이름이 있어야 하는데 없는 경우**이고, 마지막 열이 그 신호다 — 이 모델의 심볼과 값이 같다면 스코프가 그 모듈을 못 덮고 있을 수 있다. 실제로 `n_hc`(=4)가 그렇게 정수로 남아 있었다.

| 모듈 | 정수 | 축 수 | 같은 값의 심볼 |
|---|---|---|---|
| `model.layers.*.self_attn` | 5 | 338031 | — |
| `model.layers.*.block_sparse_moe.experts.*.act_fn` | 3840 | 11040 | — |
| `model.layers.*.block_sparse_moe.experts.*.act_fn` | 12 | 11040 | `n_attn_res_block` |
| `model.layers.*.self_attn` | 64 | 4278 | `d_rope`, `d_chunk` |
| `model.layers.*.self_attn` | 2 | 1173 | `E_shared` |
| `model.layers.*.self_attn` | 3 | 1173 | — |
| `model.layers.*.self_attn` | 4 | 1173 | `d_conv` |
| `model.layers.*.self_attn` | 6 | 1173 | — |
| `model.layers.*.self_attn` | 7 | 1173 | — |
| `model.layers.*.self_attn` | 8 | 1173 | — |
| `model.layers.*.self_attn` | 9 | 1173 | — |
| `model.layers.*.self_attn` | 10 | 1173 | — |
| `model.layers.*.self_attn` | 11 | 1173 | — |
| `model.layers.*.self_attn` | 12 | 1173 | `n_attn_res_block` |
| `model.layers.*.self_attn` | 13 | 1173 | — |
| `model.layers.*.self_attn` | 14 | 1173 | — |
| `model.layers.*.self_attn` | 15 | 1173 | — |
| `model.layers.*.self_attn` | 16 | 1173 | `k` |
| `model.layers.*.self_attn` | 17 | 1173 | — |
| `model.layers.*.self_attn` | 18 | 1173 | — |
| `model.layers.*.self_attn` | 19 | 1173 | — |
| `model.layers.*.self_attn` | 20 | 1173 | — |
| `model.layers.*.self_attn` | 21 | 1173 | — |
| `model.layers.*.self_attn` | 22 | 1173 | — |
| `model.layers.*.self_attn` | 23 | 1173 | — |
| `model.layers.*.self_attn` | 24 | 1173 | — |
| `model.layers.*.self_attn` | 25 | 1173 | — |
| `model.layers.*.self_attn` | 26 | 1173 | — |
| `model.layers.*.self_attn` | 27 | 1173 | — |
| `model.layers.*.self_attn` | 28 | 1173 | — |
| `model.layers.*.self_attn` | 29 | 1173 | — |
| `model.layers.*.self_attn` | 30 | 1173 | — |
| `model.layers.*.self_attn` | 31 | 1173 | — |
| `model.layers.*.self_attn` | 33 | 1173 | — |
| `model.layers.*.self_attn` | 34 | 1173 | — |
| `model.layers.*.self_attn` | 35 | 1173 | — |
| `model.layers.*.self_attn` | 36 | 1173 | — |
| `model.layers.*.self_attn` | 37 | 1173 | — |
| `model.layers.*.self_attn` | 38 | 1173 | — |
| `model.layers.*.self_attn` | 39 | 1173 | — |
| `model.layers.*.self_attn` | 40 | 1173 | — |
| `model.layers.*.self_attn` | 41 | 1173 | — |
| `model.layers.*.self_attn` | 42 | 1173 | — |
| `model.layers.*.self_attn` | 43 | 1173 | — |
| `model.layers.*.self_attn` | 44 | 1173 | — |
| `model.layers.*.self_attn` | 45 | 1173 | — |
| `model.layers.*.self_attn` | 46 | 1173 | — |
| `model.layers.*.self_attn` | 47 | 1173 | — |
| `model.layers.*.self_attn` | 48 | 1173 | — |
| `model.layers.*.self_attn` | 49 | 1173 | — |
| `model.layers.*.self_attn` | 50 | 1173 | — |
| `model.layers.*.self_attn` | 51 | 1173 | — |
| `model.layers.*.self_attn` | 52 | 1173 | — |
| `model.layers.*.self_attn` | 53 | 1173 | — |
| `model.layers.*.self_attn` | 54 | 1173 | — |
| `model.layers.*.self_attn` | 55 | 1173 | — |
| `model.layers.*.self_attn` | 56 | 1173 | — |
| `model.layers.*.self_attn` | 57 | 1173 | — |
| `model.layers.*.self_attn` | 58 | 1173 | — |
| `model.layers.*.self_attn` | 59 | 1173 | — |
| `model.layers.*.self_attn` | 60 | 1173 | — |
| `model.layers.*.self_attn` | 61 | 1173 | — |
| `model.layers.*.self_attn` | 62 | 1173 | — |
| `model.layers.*.self_attn` | 63 | 1173 | — |
| `model.layers.*.block_sparse_moe` | 3840 | 736 | — |
| `model.layers.*.block_sparse_moe.experts.*.w1` | 3840 | 736 | — |
| `model.layers.*.block_sparse_moe.experts.*.w3` | 3840 | 736 | — |
| `model.layers.*.block_sparse_moe.experts.*.w2` | 3840 | 736 | — |
| `model.layers.*.block_sparse_moe` | 12 | 736 | `n_attn_res_block` |
| `model.layers.*.block_sparse_moe.experts.*.w1` | 12 | 736 | `n_attn_res_block` |
| `model.layers.*.block_sparse_moe.experts.*.w3` | 12 | 736 | `n_attn_res_block` |
| `model.layers.*.block_sparse_moe.experts.*.w2` | 12 | 736 | `n_attn_res_block` |
| `model.layers.*.self_attn` | 32 | 414 | — |
| `model.layers.*.block_sparse_moe.experts.0` | 3840 | 276 | — |
| `model.layers.*.block_sparse_moe.experts.1` | 3840 | 276 | — |
| `model.layers.*.block_sparse_moe.experts.2` | 3840 | 276 | — |
| `model.layers.*.block_sparse_moe.experts.3` | 3840 | 276 | — |
| `model.layers.*.block_sparse_moe.experts.0` | 12 | 276 | `n_attn_res_block` |
| `model.layers.*.block_sparse_moe.experts.1` | 12 | 276 | `n_attn_res_block` |
| `model.layers.*.block_sparse_moe.experts.2` | 12 | 276 | `n_attn_res_block` |
| `model.layers.*.block_sparse_moe.experts.3` | 12 | 276 | `n_attn_res_block` |
| `model.layers.*.self_attn.q_conv1d` | 3 | 207 | — |
| `model.layers.*.self_attn.k_conv1d` | 3 | 207 | — |
| `model.layers.*.self_attn.v_conv1d` | 3 | 207 | — |
| `model.layers.1` | 2 | 128 | `E_shared` |
| `model.layers.2` | 2 | 128 | `E_shared` |
| `model.layers.3` | 2 | 128 | `E_shared` |
| `model.layers.4` | 2 | 128 | `E_shared` |
| `model.layers.5` | 2 | 128 | `E_shared` |
| `model.layers.6` | 2 | 128 | `E_shared` |
| `model.layers.7` | 2 | 128 | `E_shared` |
| `model.layers.8` | 2 | 128 | `E_shared` |
| `model.layers.9` | 2 | 128 | `E_shared` |
| `model.layers.10` | 2 | 128 | `E_shared` |
| `model.layers.11` | 2 | 128 | `E_shared` |
| `model.layers.13` | 3 | 128 | — |
| `model.layers.14` | 3 | 128 | — |
| `model.layers.15` | 3 | 128 | — |
| `model.layers.16` | 3 | 128 | — |
| `model.layers.17` | 3 | 128 | — |
| `model.layers.18` | 3 | 128 | — |
| `model.layers.19` | 3 | 128 | — |
| `model.layers.20` | 3 | 128 | — |
| `model.layers.21` | 3 | 128 | — |
| `model.layers.22` | 3 | 128 | — |
| `model.layers.23` | 3 | 128 | — |
| `model.layers.25` | 4 | 128 | `d_conv` |
| `model.layers.26` | 4 | 128 | `d_conv` |
| `model.layers.27` | 4 | 128 | `d_conv` |
| `model.layers.28` | 4 | 128 | `d_conv` |
| `model.layers.29` | 4 | 128 | `d_conv` |
| `model.layers.30` | 4 | 128 | `d_conv` |
| `model.layers.31` | 4 | 128 | `d_conv` |
| `model.layers.32` | 4 | 128 | `d_conv` |
| `model.layers.33` | 4 | 128 | `d_conv` |
| `model.layers.34` | 4 | 128 | `d_conv` |
| `model.layers.35` | 4 | 128 | `d_conv` |
| `model.layers.37` | 5 | 128 | — |
| `model.layers.38` | 5 | 128 | — |
| `model.layers.39` | 5 | 128 | — |
| `model.layers.40` | 5 | 128 | — |
| `model.layers.41` | 5 | 128 | — |
| `model.layers.42` | 5 | 128 | — |
| `model.layers.43` | 5 | 128 | — |
| `model.layers.44` | 5 | 128 | — |
| `model.layers.45` | 5 | 128 | — |
| `model.layers.46` | 5 | 128 | — |
| `model.layers.47` | 5 | 128 | — |
| `model.layers.49` | 6 | 128 | — |
| `model.layers.50` | 6 | 128 | — |
| `model.layers.51` | 6 | 128 | — |
| `model.layers.52` | 6 | 128 | — |
| `model.layers.53` | 6 | 128 | — |
| `model.layers.54` | 6 | 128 | — |
| `model.layers.55` | 6 | 128 | — |
| `model.layers.56` | 6 | 128 | — |
| `model.layers.57` | 6 | 128 | — |
| `model.layers.58` | 6 | 128 | — |
| `model.layers.59` | 6 | 128 | — |
| `model.layers.61` | 7 | 128 | — |
| `model.layers.62` | 7 | 128 | — |
| `model.layers.63` | 7 | 128 | — |
| `model.layers.64` | 7 | 128 | — |
| `model.layers.65` | 7 | 128 | — |
| `model.layers.66` | 7 | 128 | — |
| `model.layers.67` | 7 | 128 | — |
| `model.layers.68` | 7 | 128 | — |
| `model.layers.69` | 7 | 128 | — |
| `model.layers.70` | 7 | 128 | — |
| `model.layers.71` | 7 | 128 | — |
| `model.layers.73` | 8 | 128 | — |
| `model.layers.74` | 8 | 128 | — |
| `model.layers.75` | 8 | 128 | — |
| `model.layers.76` | 8 | 128 | — |
| `model.layers.77` | 8 | 128 | — |
| `model.layers.78` | 8 | 128 | — |
| `model.layers.79` | 8 | 128 | — |
| `model.layers.80` | 8 | 128 | — |
| `model.layers.81` | 8 | 128 | — |
| `model.layers.82` | 8 | 128 | — |
| `model.layers.83` | 8 | 128 | — |
| `model.layers.85` | 9 | 128 | — |
| `model.layers.86` | 9 | 128 | — |
| `model.layers.87` | 9 | 128 | — |
| `model.layers.88` | 9 | 128 | — |
| `model.layers.89` | 9 | 128 | — |
| `model.layers.90` | 9 | 128 | — |
| `model.layers.91` | 9 | 128 | — |
| `model.layers.92` | 9 | 128 | — |
| `model.layers.12` | 2 | 68 | `E_shared` |
| `model.layers.24` | 3 | 68 | — |
| `model.layers.36` | 4 | 68 | `d_conv` |
| `model.layers.48` | 5 | 68 | — |
| `model.layers.60` | 6 | 68 | — |
| `model.layers.72` | 7 | 68 | — |
| `model.layers.84` | 8 | 68 | — |
| `model.layers.0` | 2 | 64 | `E_shared` |
| `model.layers.12` | 3 | 64 | — |
| `model.layers.24` | 4 | 64 | `d_conv` |
| `model.layers.36` | 5 | 64 | — |
| `model.layers.48` | 6 | 64 | — |
| `model.layers.60` | 7 | 64 | — |
| `model.layers.72` | 8 | 64 | — |
| `model.layers.84` | 9 | 64 | — |
| `model` | 9 | 64 | — |
| `model.layers.13` | 2 | 4 | `E_shared` |
| `model.layers.14` | 2 | 4 | `E_shared` |
| `model.layers.15` | 2 | 4 | `E_shared` |
| `model.layers.16` | 2 | 4 | `E_shared` |
| `model.layers.17` | 2 | 4 | `E_shared` |
| `model.layers.18` | 2 | 4 | `E_shared` |
| `model.layers.19` | 2 | 4 | `E_shared` |
| `model.layers.20` | 2 | 4 | `E_shared` |
| `model.layers.21` | 2 | 4 | `E_shared` |
| `model.layers.22` | 2 | 4 | `E_shared` |
| `model.layers.23` | 2 | 4 | `E_shared` |
| `model.layers.24` | 2 | 4 | `E_shared` |
| `model.layers.25` | 3 | 4 | — |
| `model.layers.26` | 3 | 4 | — |
| `model.layers.27` | 3 | 4 | — |
| `model.layers.28` | 3 | 4 | — |
| `model.layers.29` | 3 | 4 | — |
| `model.layers.30` | 3 | 4 | — |
| `model.layers.31` | 3 | 4 | — |
| `model.layers.32` | 3 | 4 | — |
| `model.layers.33` | 3 | 4 | — |
| `model.layers.34` | 3 | 4 | — |
| `model.layers.35` | 3 | 4 | — |
| `model.layers.36` | 3 | 4 | — |
| `model.layers.37` | 4 | 4 | `d_conv` |
| `model.layers.38` | 4 | 4 | `d_conv` |
| `model.layers.39` | 4 | 4 | `d_conv` |
| `model.layers.40` | 4 | 4 | `d_conv` |
| `model.layers.41` | 4 | 4 | `d_conv` |
| `model.layers.42` | 4 | 4 | `d_conv` |
| `model.layers.43` | 4 | 4 | `d_conv` |
| `model.layers.44` | 4 | 4 | `d_conv` |
| `model.layers.45` | 4 | 4 | `d_conv` |
| `model.layers.46` | 4 | 4 | `d_conv` |
| `model.layers.47` | 4 | 4 | `d_conv` |
| `model.layers.48` | 4 | 4 | `d_conv` |
| `model.layers.49` | 5 | 4 | — |
| `model.layers.50` | 5 | 4 | — |
| `model.layers.51` | 5 | 4 | — |
| `model.layers.52` | 5 | 4 | — |
| `model.layers.53` | 5 | 4 | — |
| `model.layers.54` | 5 | 4 | — |
| `model.layers.55` | 5 | 4 | — |
| `model.layers.56` | 5 | 4 | — |
| `model.layers.57` | 5 | 4 | — |
| `model.layers.58` | 5 | 4 | — |
| `model.layers.59` | 5 | 4 | — |
| `model.layers.60` | 5 | 4 | — |
| `model.layers.61` | 6 | 4 | — |
| `model.layers.62` | 6 | 4 | — |
| `model.layers.63` | 6 | 4 | — |
| `model.layers.64` | 6 | 4 | — |
| `model.layers.65` | 6 | 4 | — |
| `model.layers.66` | 6 | 4 | — |
| `model.layers.67` | 6 | 4 | — |
| `model.layers.68` | 6 | 4 | — |
| `model.layers.69` | 6 | 4 | — |
| `model.layers.70` | 6 | 4 | — |
| `model.layers.71` | 6 | 4 | — |
| `model.layers.72` | 6 | 4 | — |
| `model.layers.73` | 7 | 4 | — |
| `model.layers.74` | 7 | 4 | — |
| `model.layers.75` | 7 | 4 | — |
| `model.layers.76` | 7 | 4 | — |
| `model.layers.77` | 7 | 4 | — |
| `model.layers.78` | 7 | 4 | — |
| `model.layers.79` | 7 | 4 | — |
| `model.layers.80` | 7 | 4 | — |
| `model.layers.81` | 7 | 4 | — |
| `model.layers.82` | 7 | 4 | — |
| `model.layers.83` | 7 | 4 | — |
| `model.layers.84` | 7 | 4 | — |
| `model.layers.85` | 8 | 4 | — |
| `model.layers.86` | 8 | 4 | — |
| `model.layers.87` | 8 | 4 | — |
| `model.layers.88` | 8 | 4 | — |
| `model.layers.89` | 8 | 4 | — |
| `model.layers.90` | 8 | 4 | — |
| `model.layers.91` | 8 | 4 | — |
| `model.layers.92` | 8 | 4 | — |
| `model` | 8 | 2 | — |

### C. 모듈이 내는 출력 shape 전부 (144개 모듈 / 1889종)

모듈 하나가 어떤 모양을 내놓는지 전부 적었다. 어떤 모듈에 **있을 수 없는 이름**이 섞여 있는지 보는 자리다(예: attention head 수가 Mamba mixer 안에, 전문가 수가 self_attn 안에).

- `lm_head`
  - `[[B*T, V]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, V]]`
  - `[[B, T, V]]`
  - `[[B, V]]`
  - `[[B, d_model]]`
  - `[[d_model, V]]`
- `model`
  - `[[1, 1, 1, 1]]`
  - `[[1, 1, 1, T+1]]`
  - `[[1, 1, 1, T]]`
  - `[[1, 1, 1]]`
  - `[[1, 1, T+1]]`
  - `[[1, 1, T, 1]]`
  - `[[1, 1, T, T]]`
  - `[[1, 1, T]]`
  - `[[1, 1]]`
  - `[[1, T+1]]`
  - `[[1, T]]`
  - `[[1]]`
  - `[[B*T, 0, d_model]]`
  - `[[B*T, 1, 9]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 9, 1]]`
  - `[[B*T, 9, d_model]]`
  - `[[B*T, 9]]`
  - `[[B*T, d_model]]`
  - `[[B, 0, d_model]]`
  - `[[B, 1, 1, 1]]`
  - `[[B, 1, 1, T+1]]`
  - `[[B, 1, 1]]`
  - `[[B, 1, 9]]`
  - `[[B, 1, T, T]]`
  - `[[B, 1, d_model]]`
  - `[[B, 1]]`
  - `[[B, 9, 1]]`
  - `[[B, 9, d_model]]`
  - `[[B, 9]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[B]]`
  - `[[T+1]]`
  - `[[T]]`
  - `[[]]`
  - `[[d_model]]`
  - `[]`
- `model.embed_tokens`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.*.block_sparse_moe`
  - `[[12, d_moe_lat]]`
  - `[[3840, d_moe_lat]]`
  - `[[B*T, E]]`
  - `[[B*T, d_model]]`
  - `[[B*T, d_moe_lat]]`
  - `[[B*T, k, 1]]`
  - `[[B*T, k, d_moe_lat]]`
  - `[[B*k*T, d_moe_lat]]`
  - `[[B*k*T], [B*k*T]]`
  - `[[B*k*T]]`
  - `[[B*k, d_moe_lat]]`
  - `[[B*k], [B*k]]`
  - `[[B*k]]`
  - `[[B, 1, d_model]]`
  - `[[B, E]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[B, d_moe_lat]]`
  - `[[B, k, 1]]`
  - `[[B, k, d_moe_lat]]`
- `model.layers.*.block_sparse_moe.experts.*.act_fn`
  - `[[12, d_moe]]`
  - `[[3840, d_moe]]`
- `model.layers.*.block_sparse_moe.experts.*.w1`
  - `[[12, d_moe]]`
  - `[[3840, d_moe]]`
  - `[[d_moe_lat, d_moe]]`
- `model.layers.*.block_sparse_moe.experts.*.w2`
  - `[[12, d_moe_lat]]`
  - `[[3840, d_moe_lat]]`
  - `[[d_moe, d_moe_lat]]`
- `model.layers.*.block_sparse_moe.experts.*.w3`
  - `[[12, d_moe]]`
  - `[[3840, d_moe]]`
  - `[[d_moe_lat, d_moe]]`
- `model.layers.*.block_sparse_moe.experts.0`
  - `[[12, 2*d_moe]]`
  - `[[3840, 2*d_moe]]`
- `model.layers.*.block_sparse_moe.experts.1`
  - `[[12, 2*d_moe]]`
  - `[[3840, 2*d_moe]]`
- `model.layers.*.block_sparse_moe.experts.2`
  - `[[12, 2*d_moe]]`
  - `[[3840, 2*d_moe]]`
- `model.layers.*.block_sparse_moe.experts.3`
  - `[[12, 2*d_moe]]`
  - `[[3840, 2*d_moe]]`
- `model.layers.*.block_sparse_moe.gate`
  - `[[1, E]]`
  - `[[B*T, 1]]`
  - `[[B*T, E]]`
  - `[[B*T, d_model]]`
  - `[[B*T, k], [B*T, k]]`
  - `[[B*T, k]]`
  - `[[B, 1]]`
  - `[[B, E]]`
  - `[[B, d_model]]`
  - `[[B, k], [B, k]]`
  - `[[B, k]]`
  - `[[E, d_model]]`
  - `[[d_model, E]]`
- `model.layers.*.block_sparse_moe.routed_expert_down_proj`
  - `[[B*T, d_moe_lat]]`
  - `[[B, d_moe_lat]]`
  - `[[d_model, d_moe_lat]]`
- `model.layers.*.block_sparse_moe.routed_expert_norm`
  - `[[B*T, 1]]`
  - `[[B*T, d_moe_lat]]`
  - `[[B, 1]]`
  - `[[B, d_moe_lat]]`
- `model.layers.*.block_sparse_moe.routed_expert_up_proj`
  - `[[B*T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_moe_lat, d_model]]`
- `model.layers.*.block_sparse_moe.shared_experts`
  - `[[B, 1, 2*E_shared*d_moe]]`
  - `[[B, T, 2*E_shared*d_moe]]`
- `model.layers.*.block_sparse_moe.shared_experts.act_fn`
  - `[[B, 1, E_shared*d_moe]]`
  - `[[B, T, E_shared*d_moe]]`
- `model.layers.*.block_sparse_moe.shared_experts.down_proj`
  - `[[B*T, E_shared*d_moe]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, d_model]]`
  - `[[B, E_shared*d_moe]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[E_shared*d_moe, d_model]]`
- `model.layers.*.block_sparse_moe.shared_experts.gate_proj`
  - `[[B*T, E_shared*d_moe]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, E_shared*d_moe]]`
  - `[[B, E_shared*d_moe]]`
  - `[[B, T, E_shared*d_moe]]`
  - `[[B, d_model]]`
  - `[[d_model, E_shared*d_moe]]`
- `model.layers.*.block_sparse_moe.shared_experts.up_proj`
  - `[[B*T, E_shared*d_moe]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, E_shared*d_moe]]`
  - `[[B, E_shared*d_moe]]`
  - `[[B, T, E_shared*d_moe]]`
  - `[[B, d_model]]`
  - `[[d_model, E_shared*d_moe]]`
- `model.layers.*.input_layernorm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
- `model.layers.*.mlp`
  - `[[B, 1, 2*d_ff]]`
  - `[[B, T, 2*d_ff]]`
- `model.layers.*.mlp.act_fn`
  - `[[B, 1, d_ff]]`
  - `[[B, T, d_ff]]`
- `model.layers.*.mlp.down_proj`
  - `[[B*T, d_ff]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_ff]]`
  - `[[B, d_model]]`
  - `[[d_ff, d_model]]`
- `model.layers.*.mlp.gate_proj`
  - `[[B*T, d_ff]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, d_ff]]`
  - `[[B, T, d_ff]]`
  - `[[B, d_ff]]`
  - `[[B, d_model]]`
  - `[[d_model, d_ff]]`
- `model.layers.*.mlp.up_proj`
  - `[[B*T, d_ff]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, d_ff]]`
  - `[[B, T, d_ff]]`
  - `[[B, d_ff]]`
  - `[[B, d_model]]`
  - `[[d_model, d_ff]]`
- `model.layers.*.post_attention_layernorm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
- `model.layers.*.self_attn`
  - `[[B*n_h, 1, T+1]]`
  - `[[B*n_h, 1, d_nope+d_rope]]`
  - `[[B*n_h, 1, d_nope]]`
  - `[[B*n_h, T+1, d_nope]]`
  - `[[B*n_h, T, T]]`
  - `[[B*n_h, T, d_nope+d_rope]]`
  - `[[B*n_h, T, d_nope]]`
  - `[[B*n_h, d_nope+d_rope, T+1]]`
  - `[[B*n_h, d_nope+d_rope, T]]`
  - `[[B*n_h_kda*n_chunk, d_chunk, 1]]`
  - `[[B*n_h_kda*n_chunk, d_chunk, d_chunk]]`
  - `[[B*n_h_kda*n_chunk, d_chunk, d_head_kda]]`
  - `[[B*n_h_kda*n_chunk, d_head_kda, 1]]`
  - `[[B*n_h_kda, 1, d_head_kda]]`
  - `[[B*n_h_kda, d_chunk, 1]]`
  - `[[B*n_h_kda, d_chunk, d_chunk]]`
  - `[[B*n_h_kda, d_chunk, d_head_kda]]`
  - `[[B*n_h_kda, d_head_kda, 1]]`
  - `[[B*n_h_kda, d_head_kda, d_chunk]]`
  - `[[B*n_h_kda, d_head_kda, d_head_kda]]`
  - `[[B, 1, 1, T+1]]`
  - `[[B, 1, 1, d_rope]]`
  - `[[B, 1, T, T]]`
  - `[[B, 1, T, d_rope]]`
  - `[[B, 1, c_kv], [B, 1, d_rope]]`
  - `[[B, 1, n_h*d_v]]`
  - `[[B, 1, n_h, d_nope+d_rope]]`
  - `[[B, 1, n_h, d_nope+d_v]]`
  - `[[B, 1, n_h, d_v]]`
  - `[[B, 1, n_h_kda*d_head_kda]]`
  - `[[B, 1, n_h_kda, 1, d_head_kda]]`
  - `[[B, 1, n_h_kda, 1]]`
  - `[[B, 1, n_h_kda, d_head_kda]]`
  - `[[B, 1, n_h_kda]]`
  - `[[B, 5, d_chunk, n_h_kda, d_head_kda]]`
  - `[[B, 5, d_chunk, n_h_kda]]`
  - `[[B, T, c_kv], [B, T, d_rope]]`
  - `[[B, T, n_h*d_v]]`
  - `[[B, T, n_h, d_nope+d_rope]]`
  - `[[B, T, n_h, d_nope+d_v]]`
  - `[[B, T, n_h, d_v]]`
  - `[[B, T, n_h_kda*d_head_kda]]`
  - `[[B, T, n_h_kda, 1]]`
  - `[[B, T, n_h_kda, d_head_kda]]`
  - `[[B, T, n_h_kda]]`
  - `[[B, n_chunk, d_chunk, n_h_kda, d_head_kda]]`
  - `[[B, n_h, 1, 1, T+1]]`
  - `[[B, n_h, 1, 1, d_nope+d_rope]]`
  - `[[B, n_h, 1, 1, d_nope]]`
  - `[[B, n_h, 1, T+1, 1]]`
  - `[[B, n_h, 1, T+1, d_nope+d_rope]]`
  - `[[B, n_h, 1, T+1]]`
  - `[[B, n_h, 1, T, d_nope+d_rope]]`
  - `[[B, n_h, 1, d_nope+d_rope, 1]]`
  - `[[B, n_h, 1, d_nope+d_rope]]`
  - `[[B, n_h, 1, d_nope+d_v]]`
  - `[[B, n_h, 1, d_nope, T+1]]`
  - `[[B, n_h, 1, d_nope, T]]`
  - `[[B, n_h, 1, d_nope], [B, n_h, 1, d_rope]]`
  - `[[B, n_h, 1, d_nope], [B, n_h, 1, d_v]]`
  - `[[B, n_h, 1, d_rope]]`
  - `[[B, n_h, 1, d_v, 1]]`
  - `[[B, n_h, 1, d_v]]`
  - `[[B, n_h, T+1, 1, 1]]`
  - `[[B, n_h, T+1, 1, d_nope]]`
  - `[[B, n_h, T+1, d_nope+d_rope, 1]]`
  - `[[B, n_h, T+1, d_nope+d_rope]]`
  - `[[B, n_h, T+1, d_nope, 1]]`
  - `[[B, n_h, T+1, d_nope]]`
  - `[[B, n_h, T, 1, T]]`
  - `[[B, n_h, T, 1, d_nope+d_rope]]`
  - `[[B, n_h, T, 1, d_nope]]`
  - `[[B, n_h, T, T, 1]]`
  - `[[B, n_h, T, T]]`
  - `[[B, n_h, T, d_nope+d_rope, 1]]`
  - `[[B, n_h, T, d_nope+d_rope]]`
  - `[[B, n_h, T, d_nope+d_v]]`
  - `[[B, n_h, T, d_nope, 1]]`
  - `[[B, n_h, T, d_nope], [B, n_h, T, d_rope]]`
  - `[[B, n_h, T, d_nope], [B, n_h, T, d_v]]`
  - `[[B, n_h, T, d_rope]]`
  - `[[B, n_h, T, d_v, 1]]`
  - `[[B, n_h, T, d_v]]`
  - `[[B, n_h, d_nope+d_rope, 1, 1]]`
  - `[[B, n_h, d_nope+d_rope, 1, T+1]]`
  - `[[B, n_h, d_nope+d_rope, T, 1]]`
  - `[[B, n_h_kda, 1, 5, d_chunk, d_head_kda]]`
  - `[[B, n_h_kda, 1, d_head_kda]]`
  - `[[B, n_h_kda, 1]]`
  - `[[B, n_h_kda, 5, 1, d_chunk]]`
  - `[[B, n_h_kda, 5, 1, d_head_kda]]`
  - `[[B, n_h_kda, 5, 10]]`
  - `[[B, n_h_kda, 5, 11]]`
  - `[[B, n_h_kda, 5, 12]]`
  - `[[B, n_h_kda, 5, 13]]`
  - `[[B, n_h_kda, 5, 14]]`
  - `[[B, n_h_kda, 5, 15]]`
  - `[[B, n_h_kda, 5, 16]]`
  - `[[B, n_h_kda, 5, 17]]`
  - `[[B, n_h_kda, 5, 18]]`
  - `[[B, n_h_kda, 5, 19]]`
  - `[[B, n_h_kda, 5, 1]]`
  - `[[B, n_h_kda, 5, 20]]`
  - `[[B, n_h_kda, 5, 21]]`
  - `[[B, n_h_kda, 5, 22]]`
  - `[[B, n_h_kda, 5, 23]]`
  - `[[B, n_h_kda, 5, 24]]`
  - `[[B, n_h_kda, 5, 25]]`
  - `[[B, n_h_kda, 5, 26]]`
  - `[[B, n_h_kda, 5, 27]]`
  - `[[B, n_h_kda, 5, 28]]`
  - `[[B, n_h_kda, 5, 29]]`
  - `[[B, n_h_kda, 5, 2]]`
  - `[[B, n_h_kda, 5, 30]]`
  - `[[B, n_h_kda, 5, 31]]`
  - `[[B, n_h_kda, 5, 33]]`
  - `[[B, n_h_kda, 5, 34]]`
  - `[[B, n_h_kda, 5, 35]]`
  - `[[B, n_h_kda, 5, 36]]`
  - `[[B, n_h_kda, 5, 37]]`
  - `[[B, n_h_kda, 5, 38]]`
  - `[[B, n_h_kda, 5, 39]]`
  - `[[B, n_h_kda, 5, 3]]`
  - `[[B, n_h_kda, 5, 40]]`
  - `[[B, n_h_kda, 5, 41]]`
  - `[[B, n_h_kda, 5, 42]]`
  - `[[B, n_h_kda, 5, 43]]`
  - `[[B, n_h_kda, 5, 44]]`
  - `[[B, n_h_kda, 5, 45]]`
  - `[[B, n_h_kda, 5, 46]]`
  - `[[B, n_h_kda, 5, 47]]`
  - `[[B, n_h_kda, 5, 48]]`
  - `[[B, n_h_kda, 5, 49]]`
  - `[[B, n_h_kda, 5, 4]]`
  - `[[B, n_h_kda, 5, 50]]`
  - `[[B, n_h_kda, 5, 51]]`
  - `[[B, n_h_kda, 5, 52]]`
  - `[[B, n_h_kda, 5, 53]]`
  - `[[B, n_h_kda, 5, 54]]`
  - `[[B, n_h_kda, 5, 55]]`
  - `[[B, n_h_kda, 5, 56]]`
  - `[[B, n_h_kda, 5, 57]]`
  - `[[B, n_h_kda, 5, 58]]`
  - `[[B, n_h_kda, 5, 59]]`
  - `[[B, n_h_kda, 5, 5]]`
  - `[[B, n_h_kda, 5, 60]]`
  - `[[B, n_h_kda, 5, 61]]`
  - `[[B, n_h_kda, 5, 62]]`
  - `[[B, n_h_kda, 5, 63]]`
  - `[[B, n_h_kda, 5, 6]]`
  - `[[B, n_h_kda, 5, 7]]`
  - `[[B, n_h_kda, 5, 8]]`
  - `[[B, n_h_kda, 5, 9]]`
  - `[[B, n_h_kda, 5, d_chunk, 10]]`
  - `[[B, n_h_kda, 5, d_chunk, 11]]`
  - `[[B, n_h_kda, 5, d_chunk, 12]]`
  - `[[B, n_h_kda, 5, d_chunk, 13]]`
  - `[[B, n_h_kda, 5, d_chunk, 14]]`
  - `[[B, n_h_kda, 5, d_chunk, 15]]`
  - `[[B, n_h_kda, 5, d_chunk, 16]]`
  - `[[B, n_h_kda, 5, d_chunk, 17]]`
  - `[[B, n_h_kda, 5, d_chunk, 18]]`
  - `[[B, n_h_kda, 5, d_chunk, 19]]`
  - `[[B, n_h_kda, 5, d_chunk, 1]]`
  - `[[B, n_h_kda, 5, d_chunk, 20]]`
  - `[[B, n_h_kda, 5, d_chunk, 21]]`
  - `[[B, n_h_kda, 5, d_chunk, 22]]`
  - `[[B, n_h_kda, 5, d_chunk, 23]]`
  - `[[B, n_h_kda, 5, d_chunk, 24]]`
  - `[[B, n_h_kda, 5, d_chunk, 25]]`
  - `[[B, n_h_kda, 5, d_chunk, 26]]`
  - `[[B, n_h_kda, 5, d_chunk, 27]]`
  - `[[B, n_h_kda, 5, d_chunk, 28]]`
  - `[[B, n_h_kda, 5, d_chunk, 29]]`
  - `[[B, n_h_kda, 5, d_chunk, 2]]`
  - `[[B, n_h_kda, 5, d_chunk, 30]]`
  - `[[B, n_h_kda, 5, d_chunk, 31]]`
  - `[[B, n_h_kda, 5, d_chunk, 32]]`
  - `[[B, n_h_kda, 5, d_chunk, 33]]`
  - `[[B, n_h_kda, 5, d_chunk, 34]]`
  - `[[B, n_h_kda, 5, d_chunk, 35]]`
  - `[[B, n_h_kda, 5, d_chunk, 36]]`
  - `[[B, n_h_kda, 5, d_chunk, 37]]`
  - `[[B, n_h_kda, 5, d_chunk, 38]]`
  - `[[B, n_h_kda, 5, d_chunk, 39]]`
  - `[[B, n_h_kda, 5, d_chunk, 3]]`
  - `[[B, n_h_kda, 5, d_chunk, 40]]`
  - `[[B, n_h_kda, 5, d_chunk, 41]]`
  - `[[B, n_h_kda, 5, d_chunk, 42]]`
  - `[[B, n_h_kda, 5, d_chunk, 43]]`
  - `[[B, n_h_kda, 5, d_chunk, 44]]`
  - `[[B, n_h_kda, 5, d_chunk, 45]]`
  - `[[B, n_h_kda, 5, d_chunk, 46]]`
  - `[[B, n_h_kda, 5, d_chunk, 47]]`
  - `[[B, n_h_kda, 5, d_chunk, 48]]`
  - `[[B, n_h_kda, 5, d_chunk, 49]]`
  - `[[B, n_h_kda, 5, d_chunk, 4]]`
  - `[[B, n_h_kda, 5, d_chunk, 50]]`
  - `[[B, n_h_kda, 5, d_chunk, 51]]`
  - `[[B, n_h_kda, 5, d_chunk, 52]]`
  - `[[B, n_h_kda, 5, d_chunk, 53]]`
  - `[[B, n_h_kda, 5, d_chunk, 54]]`
  - `[[B, n_h_kda, 5, d_chunk, 55]]`
  - `[[B, n_h_kda, 5, d_chunk, 56]]`
  - `[[B, n_h_kda, 5, d_chunk, 57]]`
  - `[[B, n_h_kda, 5, d_chunk, 58]]`
  - `[[B, n_h_kda, 5, d_chunk, 59]]`
  - `[[B, n_h_kda, 5, d_chunk, 5]]`
  - `[[B, n_h_kda, 5, d_chunk, 60]]`
  - `[[B, n_h_kda, 5, d_chunk, 61]]`
  - `[[B, n_h_kda, 5, d_chunk, 62]]`
  - `[[B, n_h_kda, 5, d_chunk, 63]]`
  - `[[B, n_h_kda, 5, d_chunk, 6]]`
  - `[[B, n_h_kda, 5, d_chunk, 7]]`
  - `[[B, n_h_kda, 5, d_chunk, 8]]`
  - `[[B, n_h_kda, 5, d_chunk, 9]]`
  - `[[B, n_h_kda, 5, d_chunk, d_chunk]]`
  - `[[B, n_h_kda, 5, d_chunk, d_head_kda]]`
  - `[[B, n_h_kda, 5, d_chunk]]`
  - `[[B, n_h_kda, 5, d_head_kda, 1]]`
  - `[[B, n_h_kda, 5, d_head_kda]]`
  - `[[B, n_h_kda, 5, d_rope/2]]`
  - `[[B, n_h_kda, d_chunk, 1]]`
  - `[[B, n_h_kda, d_chunk, d_chunk]]`
  - `[[B, n_h_kda, d_chunk, d_head_kda]]`
  - `[[B, n_h_kda, d_chunk]]`
  - `[[B, n_h_kda, d_head_kda, 1]]`
  - `[[B, n_h_kda, d_head_kda, d_chunk]]`
  - `[[B, n_h_kda, d_head_kda, d_head_kda]]`
  - `[[B, n_h_kda, d_head_kda]]`
  - `[[B, n_h_kda]]`
  - `[[d_chunk, d_chunk]]`
  - `[[n_h_kda, 1]]`
  - `[[n_h_kda, d_head_kda]]`
- `model.layers.*.self_attn.b_proj`
  - `[[B*T, d_model]]`
  - `[[B*T, n_h_kda]]`
  - `[[B, 1, n_h_kda]]`
  - `[[B, T, n_h_kda]]`
  - `[[B, d_model]]`
  - `[[B, n_h_kda]]`
  - `[[d_model, n_h_kda]]`
- `model.layers.*.self_attn.f_a_proj`
  - `[[B*T, d_head_kda]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, d_head_kda]]`
  - `[[B, T, d_head_kda]]`
  - `[[B, d_head_kda]]`
  - `[[B, d_model]]`
  - `[[d_model, d_head_kda]]`
- `model.layers.*.self_attn.f_b_proj`
  - `[[B*T, d_head_kda]]`
  - `[[B*T, n_h_kda*d_head_kda]]`
  - `[[B, 1, n_h_kda*d_head_kda]]`
  - `[[B, T, n_h_kda*d_head_kda]]`
  - `[[B, d_head_kda]]`
  - `[[B, n_h_kda*d_head_kda]]`
  - `[[d_head_kda, n_h_kda*d_head_kda]]`
- `model.layers.*.self_attn.g_proj`
  - `[[B*T, d_model]]`
  - `[[B*T, n_h*d_v]]`
  - `[[B*T, n_h_kda*d_head_kda]]`
  - `[[B, 1, n_h*d_v]]`
  - `[[B, 1, n_h_kda*d_head_kda]]`
  - `[[B, T, n_h*d_v]]`
  - `[[B, T, n_h_kda*d_head_kda]]`
  - `[[B, d_model]]`
  - `[[B, n_h*d_v]]`
  - `[[B, n_h_kda*d_head_kda]]`
  - `[[d_model, n_h*d_v]]`
  - `[[d_model, n_h_kda*d_head_kda]]`
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
  - `[[B*T, d_model]]`
  - `[[B*T, n_h_kda*d_head_kda]]`
  - `[[B, 1, n_h_kda*d_head_kda]]`
  - `[[B, T, n_h_kda*d_head_kda]]`
  - `[[B, d_model]]`
  - `[[B, n_h_kda*d_head_kda]]`
  - `[[d_model, n_h_kda*d_head_kda]]`
- `model.layers.*.self_attn.kv_a_layernorm`
  - `[[B, 1, 1]]`
  - `[[B, 1, c_kv]]`
  - `[[B, T, 1]]`
  - `[[B, T, c_kv]]`
- `model.layers.*.self_attn.kv_a_proj_with_mqa`
  - `[[B*T, c_kv+d_rope]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, c_kv+d_rope]]`
  - `[[B, T, c_kv+d_rope]]`
  - `[[B, c_kv+d_rope]]`
  - `[[B, d_model]]`
  - `[[d_model, c_kv+d_rope]]`
- `model.layers.*.self_attn.kv_b_proj`
  - `[[B*T, c_kv]]`
  - `[[B*T, n_h*(d_nope+d_v)]]`
  - `[[B, 1, n_h*(d_nope+d_v)]]`
  - `[[B, T, n_h*(d_nope+d_v)]]`
  - `[[B, c_kv]]`
  - `[[B, n_h*(d_nope+d_v)]]`
  - `[[c_kv, n_h*(d_nope+d_v)]]`
- `model.layers.*.self_attn.o_norm`
  - `[[B, 1, n_h_kda, 1]]`
  - `[[B, 1, n_h_kda, d_head_kda]]`
  - `[[B, T, n_h_kda, 1]]`
  - `[[B, T, n_h_kda, d_head_kda]]`
- `model.layers.*.self_attn.o_proj`
  - `[[B*T, d_model]]`
  - `[[B*T, n_h*d_v]]`
  - `[[B*T, n_h_kda*d_head_kda]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, T, n_h_kda*d_head_kda]]`
  - `[[B, d_model]]`
  - `[[B, n_h*d_v]]`
  - `[[B, n_h_kda*d_head_kda]]`
  - `[[n_h*d_v, d_model]]`
  - `[[n_h_kda*d_head_kda, d_model]]`
- `model.layers.*.self_attn.q_a_layernorm`
  - `[[B, 1, 1]]`
  - `[[B, 1, c_q]]`
  - `[[B, T, 1]]`
  - `[[B, T, c_q]]`
- `model.layers.*.self_attn.q_a_proj`
  - `[[B*T, c_q]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, c_q]]`
  - `[[B, T, c_q]]`
  - `[[B, c_q]]`
  - `[[B, d_model]]`
  - `[[d_model, c_q]]`
- `model.layers.*.self_attn.q_b_proj`
  - `[[B*T, c_q]]`
  - `[[B*T, n_h*(d_nope+d_rope)]]`
  - `[[B, 1, n_h*(d_nope+d_rope)]]`
  - `[[B, T, n_h*(d_nope+d_rope)]]`
  - `[[B, c_q]]`
  - `[[B, n_h*(d_nope+d_rope)]]`
  - `[[c_q, n_h*(d_nope+d_rope)]]`
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
  - `[[B*T, d_model]]`
  - `[[B*T, n_h_kda*d_head_kda]]`
  - `[[B, 1, n_h_kda*d_head_kda]]`
  - `[[B, T, n_h_kda*d_head_kda]]`
  - `[[B, d_model]]`
  - `[[B, n_h_kda*d_head_kda]]`
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
  - `[[B*T, d_model]]`
  - `[[B*T, n_h_kda*d_head_kda]]`
  - `[[B, 1, n_h_kda*d_head_kda]]`
  - `[[B, T, n_h_kda*d_head_kda]]`
  - `[[B, d_model]]`
  - `[[B, n_h_kda*d_head_kda]]`
  - `[[d_model, n_h_kda*d_head_kda]]`
- `model.layers.0`
  - `[[B*T, 1, 2]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 2, 1]]`
  - `[[B*T, 2, d_model]]`
  - `[[B*T, 2]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 2]]`
  - `[[B, 1, d_model]]`
  - `[[B, 2, 1]]`
  - `[[B, 2, d_model]]`
  - `[[B, 2]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.1`
  - `[[B*T, 1, 2]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 2, 1]]`
  - `[[B*T, 2, d_model]]`
  - `[[B*T, 2]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 2]]`
  - `[[B, 1, d_model]]`
  - `[[B, 2, 1]]`
  - `[[B, 2, d_model]]`
  - `[[B, 2]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.10`
  - `[[B*T, 1, 2]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 2, 1]]`
  - `[[B*T, 2, d_model]]`
  - `[[B*T, 2]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 2]]`
  - `[[B, 1, d_model]]`
  - `[[B, 2, 1]]`
  - `[[B, 2, d_model]]`
  - `[[B, 2]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.11`
  - `[[B*T, 1, 2]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 2, 1]]`
  - `[[B*T, 2, d_model]]`
  - `[[B*T, 2]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 2]]`
  - `[[B, 1, d_model]]`
  - `[[B, 2, 1]]`
  - `[[B, 2, d_model]]`
  - `[[B, 2]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.12`
  - `[[B*T, 1, 2]]`
  - `[[B*T, 1, 3]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 2, 1]]`
  - `[[B*T, 2, d_model]]`
  - `[[B*T, 2]]`
  - `[[B*T, 3, 1]]`
  - `[[B*T, 3, d_model]]`
  - `[[B*T, 3]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 2]]`
  - `[[B, 1, 3]]`
  - `[[B, 1, d_model]]`
  - `[[B, 2, 1]]`
  - `[[B, 2, d_model]]`
  - `[[B, 2]]`
  - `[[B, 3, 1]]`
  - `[[B, 3, d_model]]`
  - `[[B, 3]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.13`
  - `[[B*T, 1, 3]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 3, 1]]`
  - `[[B*T, 3, d_model]]`
  - `[[B*T, 3]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 3]]`
  - `[[B, 1, d_model]]`
  - `[[B, 3, 1]]`
  - `[[B, 3, d_model]]`
  - `[[B, 3]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.14`
  - `[[B*T, 1, 3]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 3, 1]]`
  - `[[B*T, 3, d_model]]`
  - `[[B*T, 3]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 3]]`
  - `[[B, 1, d_model]]`
  - `[[B, 3, 1]]`
  - `[[B, 3, d_model]]`
  - `[[B, 3]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.15`
  - `[[B*T, 1, 3]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 3, 1]]`
  - `[[B*T, 3, d_model]]`
  - `[[B*T, 3]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 3]]`
  - `[[B, 1, d_model]]`
  - `[[B, 3, 1]]`
  - `[[B, 3, d_model]]`
  - `[[B, 3]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.16`
  - `[[B*T, 1, 3]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 3, 1]]`
  - `[[B*T, 3, d_model]]`
  - `[[B*T, 3]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 3]]`
  - `[[B, 1, d_model]]`
  - `[[B, 3, 1]]`
  - `[[B, 3, d_model]]`
  - `[[B, 3]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.17`
  - `[[B*T, 1, 3]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 3, 1]]`
  - `[[B*T, 3, d_model]]`
  - `[[B*T, 3]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 3]]`
  - `[[B, 1, d_model]]`
  - `[[B, 3, 1]]`
  - `[[B, 3, d_model]]`
  - `[[B, 3]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.18`
  - `[[B*T, 1, 3]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 3, 1]]`
  - `[[B*T, 3, d_model]]`
  - `[[B*T, 3]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 3]]`
  - `[[B, 1, d_model]]`
  - `[[B, 3, 1]]`
  - `[[B, 3, d_model]]`
  - `[[B, 3]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.19`
  - `[[B*T, 1, 3]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 3, 1]]`
  - `[[B*T, 3, d_model]]`
  - `[[B*T, 3]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 3]]`
  - `[[B, 1, d_model]]`
  - `[[B, 3, 1]]`
  - `[[B, 3, d_model]]`
  - `[[B, 3]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.2`
  - `[[B*T, 1, 2]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 2, 1]]`
  - `[[B*T, 2, d_model]]`
  - `[[B*T, 2]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 2]]`
  - `[[B, 1, d_model]]`
  - `[[B, 2, 1]]`
  - `[[B, 2, d_model]]`
  - `[[B, 2]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.20`
  - `[[B*T, 1, 3]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 3, 1]]`
  - `[[B*T, 3, d_model]]`
  - `[[B*T, 3]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 3]]`
  - `[[B, 1, d_model]]`
  - `[[B, 3, 1]]`
  - `[[B, 3, d_model]]`
  - `[[B, 3]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.21`
  - `[[B*T, 1, 3]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 3, 1]]`
  - `[[B*T, 3, d_model]]`
  - `[[B*T, 3]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 3]]`
  - `[[B, 1, d_model]]`
  - `[[B, 3, 1]]`
  - `[[B, 3, d_model]]`
  - `[[B, 3]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.22`
  - `[[B*T, 1, 3]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 3, 1]]`
  - `[[B*T, 3, d_model]]`
  - `[[B*T, 3]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 3]]`
  - `[[B, 1, d_model]]`
  - `[[B, 3, 1]]`
  - `[[B, 3, d_model]]`
  - `[[B, 3]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.23`
  - `[[B*T, 1, 3]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 3, 1]]`
  - `[[B*T, 3, d_model]]`
  - `[[B*T, 3]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 3]]`
  - `[[B, 1, d_model]]`
  - `[[B, 3, 1]]`
  - `[[B, 3, d_model]]`
  - `[[B, 3]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.24`
  - `[[B*T, 1, 3]]`
  - `[[B*T, 1, 4]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 3, 1]]`
  - `[[B*T, 3, d_model]]`
  - `[[B*T, 3]]`
  - `[[B*T, 4, 1]]`
  - `[[B*T, 4, d_model]]`
  - `[[B*T, 4]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 3]]`
  - `[[B, 1, 4]]`
  - `[[B, 1, d_model]]`
  - `[[B, 3, 1]]`
  - `[[B, 3, d_model]]`
  - `[[B, 3]]`
  - `[[B, 4, 1]]`
  - `[[B, 4, d_model]]`
  - `[[B, 4]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.25`
  - `[[B*T, 1, 4]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 4, 1]]`
  - `[[B*T, 4, d_model]]`
  - `[[B*T, 4]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 4]]`
  - `[[B, 1, d_model]]`
  - `[[B, 4, 1]]`
  - `[[B, 4, d_model]]`
  - `[[B, 4]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.26`
  - `[[B*T, 1, 4]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 4, 1]]`
  - `[[B*T, 4, d_model]]`
  - `[[B*T, 4]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 4]]`
  - `[[B, 1, d_model]]`
  - `[[B, 4, 1]]`
  - `[[B, 4, d_model]]`
  - `[[B, 4]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.27`
  - `[[B*T, 1, 4]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 4, 1]]`
  - `[[B*T, 4, d_model]]`
  - `[[B*T, 4]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 4]]`
  - `[[B, 1, d_model]]`
  - `[[B, 4, 1]]`
  - `[[B, 4, d_model]]`
  - `[[B, 4]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.28`
  - `[[B*T, 1, 4]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 4, 1]]`
  - `[[B*T, 4, d_model]]`
  - `[[B*T, 4]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 4]]`
  - `[[B, 1, d_model]]`
  - `[[B, 4, 1]]`
  - `[[B, 4, d_model]]`
  - `[[B, 4]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.29`
  - `[[B*T, 1, 4]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 4, 1]]`
  - `[[B*T, 4, d_model]]`
  - `[[B*T, 4]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 4]]`
  - `[[B, 1, d_model]]`
  - `[[B, 4, 1]]`
  - `[[B, 4, d_model]]`
  - `[[B, 4]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.3`
  - `[[B*T, 1, 2]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 2, 1]]`
  - `[[B*T, 2, d_model]]`
  - `[[B*T, 2]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 2]]`
  - `[[B, 1, d_model]]`
  - `[[B, 2, 1]]`
  - `[[B, 2, d_model]]`
  - `[[B, 2]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.30`
  - `[[B*T, 1, 4]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 4, 1]]`
  - `[[B*T, 4, d_model]]`
  - `[[B*T, 4]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 4]]`
  - `[[B, 1, d_model]]`
  - `[[B, 4, 1]]`
  - `[[B, 4, d_model]]`
  - `[[B, 4]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.31`
  - `[[B*T, 1, 4]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 4, 1]]`
  - `[[B*T, 4, d_model]]`
  - `[[B*T, 4]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 4]]`
  - `[[B, 1, d_model]]`
  - `[[B, 4, 1]]`
  - `[[B, 4, d_model]]`
  - `[[B, 4]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.32`
  - `[[B*T, 1, 4]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 4, 1]]`
  - `[[B*T, 4, d_model]]`
  - `[[B*T, 4]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 4]]`
  - `[[B, 1, d_model]]`
  - `[[B, 4, 1]]`
  - `[[B, 4, d_model]]`
  - `[[B, 4]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.33`
  - `[[B*T, 1, 4]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 4, 1]]`
  - `[[B*T, 4, d_model]]`
  - `[[B*T, 4]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 4]]`
  - `[[B, 1, d_model]]`
  - `[[B, 4, 1]]`
  - `[[B, 4, d_model]]`
  - `[[B, 4]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.34`
  - `[[B*T, 1, 4]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 4, 1]]`
  - `[[B*T, 4, d_model]]`
  - `[[B*T, 4]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 4]]`
  - `[[B, 1, d_model]]`
  - `[[B, 4, 1]]`
  - `[[B, 4, d_model]]`
  - `[[B, 4]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.35`
  - `[[B*T, 1, 4]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 4, 1]]`
  - `[[B*T, 4, d_model]]`
  - `[[B*T, 4]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 4]]`
  - `[[B, 1, d_model]]`
  - `[[B, 4, 1]]`
  - `[[B, 4, d_model]]`
  - `[[B, 4]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.36`
  - `[[B*T, 1, 4]]`
  - `[[B*T, 1, 5]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 4, 1]]`
  - `[[B*T, 4, d_model]]`
  - `[[B*T, 4]]`
  - `[[B*T, 5, 1]]`
  - `[[B*T, 5, d_model]]`
  - `[[B*T, 5]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 4]]`
  - `[[B, 1, 5]]`
  - `[[B, 1, d_model]]`
  - `[[B, 4, 1]]`
  - `[[B, 4, d_model]]`
  - `[[B, 4]]`
  - `[[B, 5, 1]]`
  - `[[B, 5, d_model]]`
  - `[[B, 5]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.37`
  - `[[B*T, 1, 5]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 5, 1]]`
  - `[[B*T, 5, d_model]]`
  - `[[B*T, 5]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 5]]`
  - `[[B, 1, d_model]]`
  - `[[B, 5, 1]]`
  - `[[B, 5, d_model]]`
  - `[[B, 5]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.38`
  - `[[B*T, 1, 5]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 5, 1]]`
  - `[[B*T, 5, d_model]]`
  - `[[B*T, 5]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 5]]`
  - `[[B, 1, d_model]]`
  - `[[B, 5, 1]]`
  - `[[B, 5, d_model]]`
  - `[[B, 5]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.39`
  - `[[B*T, 1, 5]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 5, 1]]`
  - `[[B*T, 5, d_model]]`
  - `[[B*T, 5]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 5]]`
  - `[[B, 1, d_model]]`
  - `[[B, 5, 1]]`
  - `[[B, 5, d_model]]`
  - `[[B, 5]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.4`
  - `[[B*T, 1, 2]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 2, 1]]`
  - `[[B*T, 2, d_model]]`
  - `[[B*T, 2]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 2]]`
  - `[[B, 1, d_model]]`
  - `[[B, 2, 1]]`
  - `[[B, 2, d_model]]`
  - `[[B, 2]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.40`
  - `[[B*T, 1, 5]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 5, 1]]`
  - `[[B*T, 5, d_model]]`
  - `[[B*T, 5]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 5]]`
  - `[[B, 1, d_model]]`
  - `[[B, 5, 1]]`
  - `[[B, 5, d_model]]`
  - `[[B, 5]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.41`
  - `[[B*T, 1, 5]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 5, 1]]`
  - `[[B*T, 5, d_model]]`
  - `[[B*T, 5]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 5]]`
  - `[[B, 1, d_model]]`
  - `[[B, 5, 1]]`
  - `[[B, 5, d_model]]`
  - `[[B, 5]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.42`
  - `[[B*T, 1, 5]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 5, 1]]`
  - `[[B*T, 5, d_model]]`
  - `[[B*T, 5]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 5]]`
  - `[[B, 1, d_model]]`
  - `[[B, 5, 1]]`
  - `[[B, 5, d_model]]`
  - `[[B, 5]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.43`
  - `[[B*T, 1, 5]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 5, 1]]`
  - `[[B*T, 5, d_model]]`
  - `[[B*T, 5]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 5]]`
  - `[[B, 1, d_model]]`
  - `[[B, 5, 1]]`
  - `[[B, 5, d_model]]`
  - `[[B, 5]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.44`
  - `[[B*T, 1, 5]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 5, 1]]`
  - `[[B*T, 5, d_model]]`
  - `[[B*T, 5]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 5]]`
  - `[[B, 1, d_model]]`
  - `[[B, 5, 1]]`
  - `[[B, 5, d_model]]`
  - `[[B, 5]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.45`
  - `[[B*T, 1, 5]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 5, 1]]`
  - `[[B*T, 5, d_model]]`
  - `[[B*T, 5]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 5]]`
  - `[[B, 1, d_model]]`
  - `[[B, 5, 1]]`
  - `[[B, 5, d_model]]`
  - `[[B, 5]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.46`
  - `[[B*T, 1, 5]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 5, 1]]`
  - `[[B*T, 5, d_model]]`
  - `[[B*T, 5]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 5]]`
  - `[[B, 1, d_model]]`
  - `[[B, 5, 1]]`
  - `[[B, 5, d_model]]`
  - `[[B, 5]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.47`
  - `[[B*T, 1, 5]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 5, 1]]`
  - `[[B*T, 5, d_model]]`
  - `[[B*T, 5]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 5]]`
  - `[[B, 1, d_model]]`
  - `[[B, 5, 1]]`
  - `[[B, 5, d_model]]`
  - `[[B, 5]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.48`
  - `[[B*T, 1, 5]]`
  - `[[B*T, 1, 6]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 5, 1]]`
  - `[[B*T, 5, d_model]]`
  - `[[B*T, 5]]`
  - `[[B*T, 6, 1]]`
  - `[[B*T, 6, d_model]]`
  - `[[B*T, 6]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 5]]`
  - `[[B, 1, 6]]`
  - `[[B, 1, d_model]]`
  - `[[B, 5, 1]]`
  - `[[B, 5, d_model]]`
  - `[[B, 5]]`
  - `[[B, 6, 1]]`
  - `[[B, 6, d_model]]`
  - `[[B, 6]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.49`
  - `[[B*T, 1, 6]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 6, 1]]`
  - `[[B*T, 6, d_model]]`
  - `[[B*T, 6]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 6]]`
  - `[[B, 1, d_model]]`
  - `[[B, 6, 1]]`
  - `[[B, 6, d_model]]`
  - `[[B, 6]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.5`
  - `[[B*T, 1, 2]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 2, 1]]`
  - `[[B*T, 2, d_model]]`
  - `[[B*T, 2]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 2]]`
  - `[[B, 1, d_model]]`
  - `[[B, 2, 1]]`
  - `[[B, 2, d_model]]`
  - `[[B, 2]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.50`
  - `[[B*T, 1, 6]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 6, 1]]`
  - `[[B*T, 6, d_model]]`
  - `[[B*T, 6]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 6]]`
  - `[[B, 1, d_model]]`
  - `[[B, 6, 1]]`
  - `[[B, 6, d_model]]`
  - `[[B, 6]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.51`
  - `[[B*T, 1, 6]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 6, 1]]`
  - `[[B*T, 6, d_model]]`
  - `[[B*T, 6]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 6]]`
  - `[[B, 1, d_model]]`
  - `[[B, 6, 1]]`
  - `[[B, 6, d_model]]`
  - `[[B, 6]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.52`
  - `[[B*T, 1, 6]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 6, 1]]`
  - `[[B*T, 6, d_model]]`
  - `[[B*T, 6]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 6]]`
  - `[[B, 1, d_model]]`
  - `[[B, 6, 1]]`
  - `[[B, 6, d_model]]`
  - `[[B, 6]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.53`
  - `[[B*T, 1, 6]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 6, 1]]`
  - `[[B*T, 6, d_model]]`
  - `[[B*T, 6]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 6]]`
  - `[[B, 1, d_model]]`
  - `[[B, 6, 1]]`
  - `[[B, 6, d_model]]`
  - `[[B, 6]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.54`
  - `[[B*T, 1, 6]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 6, 1]]`
  - `[[B*T, 6, d_model]]`
  - `[[B*T, 6]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 6]]`
  - `[[B, 1, d_model]]`
  - `[[B, 6, 1]]`
  - `[[B, 6, d_model]]`
  - `[[B, 6]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.55`
  - `[[B*T, 1, 6]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 6, 1]]`
  - `[[B*T, 6, d_model]]`
  - `[[B*T, 6]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 6]]`
  - `[[B, 1, d_model]]`
  - `[[B, 6, 1]]`
  - `[[B, 6, d_model]]`
  - `[[B, 6]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.56`
  - `[[B*T, 1, 6]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 6, 1]]`
  - `[[B*T, 6, d_model]]`
  - `[[B*T, 6]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 6]]`
  - `[[B, 1, d_model]]`
  - `[[B, 6, 1]]`
  - `[[B, 6, d_model]]`
  - `[[B, 6]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.57`
  - `[[B*T, 1, 6]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 6, 1]]`
  - `[[B*T, 6, d_model]]`
  - `[[B*T, 6]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 6]]`
  - `[[B, 1, d_model]]`
  - `[[B, 6, 1]]`
  - `[[B, 6, d_model]]`
  - `[[B, 6]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.58`
  - `[[B*T, 1, 6]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 6, 1]]`
  - `[[B*T, 6, d_model]]`
  - `[[B*T, 6]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 6]]`
  - `[[B, 1, d_model]]`
  - `[[B, 6, 1]]`
  - `[[B, 6, d_model]]`
  - `[[B, 6]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.59`
  - `[[B*T, 1, 6]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 6, 1]]`
  - `[[B*T, 6, d_model]]`
  - `[[B*T, 6]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 6]]`
  - `[[B, 1, d_model]]`
  - `[[B, 6, 1]]`
  - `[[B, 6, d_model]]`
  - `[[B, 6]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.6`
  - `[[B*T, 1, 2]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 2, 1]]`
  - `[[B*T, 2, d_model]]`
  - `[[B*T, 2]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 2]]`
  - `[[B, 1, d_model]]`
  - `[[B, 2, 1]]`
  - `[[B, 2, d_model]]`
  - `[[B, 2]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.60`
  - `[[B*T, 1, 6]]`
  - `[[B*T, 1, 7]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 6, 1]]`
  - `[[B*T, 6, d_model]]`
  - `[[B*T, 6]]`
  - `[[B*T, 7, 1]]`
  - `[[B*T, 7, d_model]]`
  - `[[B*T, 7]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 6]]`
  - `[[B, 1, 7]]`
  - `[[B, 1, d_model]]`
  - `[[B, 6, 1]]`
  - `[[B, 6, d_model]]`
  - `[[B, 6]]`
  - `[[B, 7, 1]]`
  - `[[B, 7, d_model]]`
  - `[[B, 7]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.61`
  - `[[B*T, 1, 7]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 7, 1]]`
  - `[[B*T, 7, d_model]]`
  - `[[B*T, 7]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 7]]`
  - `[[B, 1, d_model]]`
  - `[[B, 7, 1]]`
  - `[[B, 7, d_model]]`
  - `[[B, 7]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.62`
  - `[[B*T, 1, 7]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 7, 1]]`
  - `[[B*T, 7, d_model]]`
  - `[[B*T, 7]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 7]]`
  - `[[B, 1, d_model]]`
  - `[[B, 7, 1]]`
  - `[[B, 7, d_model]]`
  - `[[B, 7]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.63`
  - `[[B*T, 1, 7]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 7, 1]]`
  - `[[B*T, 7, d_model]]`
  - `[[B*T, 7]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 7]]`
  - `[[B, 1, d_model]]`
  - `[[B, 7, 1]]`
  - `[[B, 7, d_model]]`
  - `[[B, 7]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.64`
  - `[[B*T, 1, 7]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 7, 1]]`
  - `[[B*T, 7, d_model]]`
  - `[[B*T, 7]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 7]]`
  - `[[B, 1, d_model]]`
  - `[[B, 7, 1]]`
  - `[[B, 7, d_model]]`
  - `[[B, 7]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.65`
  - `[[B*T, 1, 7]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 7, 1]]`
  - `[[B*T, 7, d_model]]`
  - `[[B*T, 7]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 7]]`
  - `[[B, 1, d_model]]`
  - `[[B, 7, 1]]`
  - `[[B, 7, d_model]]`
  - `[[B, 7]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.66`
  - `[[B*T, 1, 7]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 7, 1]]`
  - `[[B*T, 7, d_model]]`
  - `[[B*T, 7]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 7]]`
  - `[[B, 1, d_model]]`
  - `[[B, 7, 1]]`
  - `[[B, 7, d_model]]`
  - `[[B, 7]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.67`
  - `[[B*T, 1, 7]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 7, 1]]`
  - `[[B*T, 7, d_model]]`
  - `[[B*T, 7]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 7]]`
  - `[[B, 1, d_model]]`
  - `[[B, 7, 1]]`
  - `[[B, 7, d_model]]`
  - `[[B, 7]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.68`
  - `[[B*T, 1, 7]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 7, 1]]`
  - `[[B*T, 7, d_model]]`
  - `[[B*T, 7]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 7]]`
  - `[[B, 1, d_model]]`
  - `[[B, 7, 1]]`
  - `[[B, 7, d_model]]`
  - `[[B, 7]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.69`
  - `[[B*T, 1, 7]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 7, 1]]`
  - `[[B*T, 7, d_model]]`
  - `[[B*T, 7]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 7]]`
  - `[[B, 1, d_model]]`
  - `[[B, 7, 1]]`
  - `[[B, 7, d_model]]`
  - `[[B, 7]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.7`
  - `[[B*T, 1, 2]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 2, 1]]`
  - `[[B*T, 2, d_model]]`
  - `[[B*T, 2]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 2]]`
  - `[[B, 1, d_model]]`
  - `[[B, 2, 1]]`
  - `[[B, 2, d_model]]`
  - `[[B, 2]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.70`
  - `[[B*T, 1, 7]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 7, 1]]`
  - `[[B*T, 7, d_model]]`
  - `[[B*T, 7]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 7]]`
  - `[[B, 1, d_model]]`
  - `[[B, 7, 1]]`
  - `[[B, 7, d_model]]`
  - `[[B, 7]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.71`
  - `[[B*T, 1, 7]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 7, 1]]`
  - `[[B*T, 7, d_model]]`
  - `[[B*T, 7]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 7]]`
  - `[[B, 1, d_model]]`
  - `[[B, 7, 1]]`
  - `[[B, 7, d_model]]`
  - `[[B, 7]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.72`
  - `[[B*T, 1, 7]]`
  - `[[B*T, 1, 8]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 7, 1]]`
  - `[[B*T, 7, d_model]]`
  - `[[B*T, 7]]`
  - `[[B*T, 8, 1]]`
  - `[[B*T, 8, d_model]]`
  - `[[B*T, 8]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 7]]`
  - `[[B, 1, 8]]`
  - `[[B, 1, d_model]]`
  - `[[B, 7, 1]]`
  - `[[B, 7, d_model]]`
  - `[[B, 7]]`
  - `[[B, 8, 1]]`
  - `[[B, 8, d_model]]`
  - `[[B, 8]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.73`
  - `[[B*T, 1, 8]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 8, 1]]`
  - `[[B*T, 8, d_model]]`
  - `[[B*T, 8]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 8]]`
  - `[[B, 1, d_model]]`
  - `[[B, 8, 1]]`
  - `[[B, 8, d_model]]`
  - `[[B, 8]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.74`
  - `[[B*T, 1, 8]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 8, 1]]`
  - `[[B*T, 8, d_model]]`
  - `[[B*T, 8]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 8]]`
  - `[[B, 1, d_model]]`
  - `[[B, 8, 1]]`
  - `[[B, 8, d_model]]`
  - `[[B, 8]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.75`
  - `[[B*T, 1, 8]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 8, 1]]`
  - `[[B*T, 8, d_model]]`
  - `[[B*T, 8]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 8]]`
  - `[[B, 1, d_model]]`
  - `[[B, 8, 1]]`
  - `[[B, 8, d_model]]`
  - `[[B, 8]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.76`
  - `[[B*T, 1, 8]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 8, 1]]`
  - `[[B*T, 8, d_model]]`
  - `[[B*T, 8]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 8]]`
  - `[[B, 1, d_model]]`
  - `[[B, 8, 1]]`
  - `[[B, 8, d_model]]`
  - `[[B, 8]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.77`
  - `[[B*T, 1, 8]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 8, 1]]`
  - `[[B*T, 8, d_model]]`
  - `[[B*T, 8]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 8]]`
  - `[[B, 1, d_model]]`
  - `[[B, 8, 1]]`
  - `[[B, 8, d_model]]`
  - `[[B, 8]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.78`
  - `[[B*T, 1, 8]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 8, 1]]`
  - `[[B*T, 8, d_model]]`
  - `[[B*T, 8]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 8]]`
  - `[[B, 1, d_model]]`
  - `[[B, 8, 1]]`
  - `[[B, 8, d_model]]`
  - `[[B, 8]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.79`
  - `[[B*T, 1, 8]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 8, 1]]`
  - `[[B*T, 8, d_model]]`
  - `[[B*T, 8]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 8]]`
  - `[[B, 1, d_model]]`
  - `[[B, 8, 1]]`
  - `[[B, 8, d_model]]`
  - `[[B, 8]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.8`
  - `[[B*T, 1, 2]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 2, 1]]`
  - `[[B*T, 2, d_model]]`
  - `[[B*T, 2]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 2]]`
  - `[[B, 1, d_model]]`
  - `[[B, 2, 1]]`
  - `[[B, 2, d_model]]`
  - `[[B, 2]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.80`
  - `[[B*T, 1, 8]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 8, 1]]`
  - `[[B*T, 8, d_model]]`
  - `[[B*T, 8]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 8]]`
  - `[[B, 1, d_model]]`
  - `[[B, 8, 1]]`
  - `[[B, 8, d_model]]`
  - `[[B, 8]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.81`
  - `[[B*T, 1, 8]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 8, 1]]`
  - `[[B*T, 8, d_model]]`
  - `[[B*T, 8]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 8]]`
  - `[[B, 1, d_model]]`
  - `[[B, 8, 1]]`
  - `[[B, 8, d_model]]`
  - `[[B, 8]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.82`
  - `[[B*T, 1, 8]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 8, 1]]`
  - `[[B*T, 8, d_model]]`
  - `[[B*T, 8]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 8]]`
  - `[[B, 1, d_model]]`
  - `[[B, 8, 1]]`
  - `[[B, 8, d_model]]`
  - `[[B, 8]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.83`
  - `[[B*T, 1, 8]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 8, 1]]`
  - `[[B*T, 8, d_model]]`
  - `[[B*T, 8]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 8]]`
  - `[[B, 1, d_model]]`
  - `[[B, 8, 1]]`
  - `[[B, 8, d_model]]`
  - `[[B, 8]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.84`
  - `[[B*T, 1, 8]]`
  - `[[B*T, 1, 9]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 8, 1]]`
  - `[[B*T, 8, d_model]]`
  - `[[B*T, 8]]`
  - `[[B*T, 9, 1]]`
  - `[[B*T, 9, d_model]]`
  - `[[B*T, 9]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 8]]`
  - `[[B, 1, 9]]`
  - `[[B, 1, d_model]]`
  - `[[B, 8, 1]]`
  - `[[B, 8, d_model]]`
  - `[[B, 8]]`
  - `[[B, 9, 1]]`
  - `[[B, 9, d_model]]`
  - `[[B, 9]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.85`
  - `[[B*T, 1, 9]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 9, 1]]`
  - `[[B*T, 9, d_model]]`
  - `[[B*T, 9]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 9]]`
  - `[[B, 1, d_model]]`
  - `[[B, 9, 1]]`
  - `[[B, 9, d_model]]`
  - `[[B, 9]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.86`
  - `[[B*T, 1, 9]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 9, 1]]`
  - `[[B*T, 9, d_model]]`
  - `[[B*T, 9]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 9]]`
  - `[[B, 1, d_model]]`
  - `[[B, 9, 1]]`
  - `[[B, 9, d_model]]`
  - `[[B, 9]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.87`
  - `[[B*T, 1, 9]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 9, 1]]`
  - `[[B*T, 9, d_model]]`
  - `[[B*T, 9]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 9]]`
  - `[[B, 1, d_model]]`
  - `[[B, 9, 1]]`
  - `[[B, 9, d_model]]`
  - `[[B, 9]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.88`
  - `[[B*T, 1, 9]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 9, 1]]`
  - `[[B*T, 9, d_model]]`
  - `[[B*T, 9]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 9]]`
  - `[[B, 1, d_model]]`
  - `[[B, 9, 1]]`
  - `[[B, 9, d_model]]`
  - `[[B, 9]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.89`
  - `[[B*T, 1, 9]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 9, 1]]`
  - `[[B*T, 9, d_model]]`
  - `[[B*T, 9]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 9]]`
  - `[[B, 1, d_model]]`
  - `[[B, 9, 1]]`
  - `[[B, 9, d_model]]`
  - `[[B, 9]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.9`
  - `[[B*T, 1, 2]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 2, 1]]`
  - `[[B*T, 2, d_model]]`
  - `[[B*T, 2]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 2]]`
  - `[[B, 1, d_model]]`
  - `[[B, 2, 1]]`
  - `[[B, 2, d_model]]`
  - `[[B, 2]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.90`
  - `[[B*T, 1, 9]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 9, 1]]`
  - `[[B*T, 9, d_model]]`
  - `[[B*T, 9]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 9]]`
  - `[[B, 1, d_model]]`
  - `[[B, 9, 1]]`
  - `[[B, 9, d_model]]`
  - `[[B, 9]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.91`
  - `[[B*T, 1, 9]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 9, 1]]`
  - `[[B*T, 9, d_model]]`
  - `[[B*T, 9]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 9]]`
  - `[[B, 1, d_model]]`
  - `[[B, 9, 1]]`
  - `[[B, 9, d_model]]`
  - `[[B, 9]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.layers.92`
  - `[[B*T, 1, 9]]`
  - `[[B*T, 1, d_model]]`
  - `[[B*T, 9, 1]]`
  - `[[B*T, 9, d_model]]`
  - `[[B*T, 9]]`
  - `[[B*T, d_model]]`
  - `[[B, 1, 9]]`
  - `[[B, 1, d_model]]`
  - `[[B, 9, 1]]`
  - `[[B, 9, d_model]]`
  - `[[B, 9]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[d_model]]`
- `model.norm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`

## 이 의뢰서를 처리하는 법

`review/prompt.md` 를 LLM 에 넘기고 이 모델을 지정한다. 판정 4종과 근거 요건, 결과를 어디에 어떤 형식으로 쓰는지는 전부 `review/` 안에 있다.
