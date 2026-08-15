# 검토 의뢰서 — deepseek-ai/DeepSeek-V4-Flash

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `deepseek_v4`
- 판단 필요: **12건**

## 증거 — 이미 받아둔 실제 소스

- `develop/sources/modeling_deepseek_v4.py` — 있음, 이 파일을 열어서 판정한다
- `develop/sources/configuration_deepseek_v4.py` — 있음, 이 파일을 열어서 판정한다

- 온라인 원본: https://github.com/huggingface/transformers/tree/main/src/transformers/models/deepseek_v4

그 밖의 재료: `full/review.md`(리뷰 패킷 — shape 별 실제 행 표본), `structure.yaml`(이 모델의 심볼 표), `full/<phase>.csv`(전체 operator 표).

## 판단이 필요한 것

### 6. 값이 겹쳐 **임의로** 고른 축

두 심볼이 같은 값을 갖는 자리다. 규칙에는 고를 근거가 없고, 이긴 쪽은 전역 우선순위 — 즉 **관례**로 정해졌다. 이름이 맞을 수도 있지만 파이프라인은 그걸 알지 못한다. 표에서는 확신 있는 라벨과 똑같이 보인다.

**소스를 열어 어느 쪽인지 확정하는 것이 여기서 할 일이다.** 확정되면 `rules/label_overrides.yaml` 에 근거와 함께 못 박는다(review/05-overrides.md). 출신으로만 구별되는 경우라면 그렇게 적고 `open` 으로 남긴다.

- `d_rope vs n_h` in `model.layers.*.self_attn` — 값 64 를 두고 후보가 2개, 14706축
- `d_rope vs n_h vs n_h_I` in `model.layers.*.self_attn.compressor.indexer` — 값 64 를 두고 후보가 3개, 2436축
- `c_I vs w_local` in `model.layers.*.self_attn.compressor.indexer` — 값 128 를 두고 후보가 2개, 1491축
- `d_rope vs n_h` in `model.layers.*.self_attn.q_b_norm` — 값 64 를 두고 후보가 2개, 1290축
- `d_rope vs n_h` in `model.layers.*.self_attn.compressor` — 값 64 를 두고 후보가 2개, 1066축
- `d_rope vs n_h vs n_h_I` in `model.layers.*.self_attn.compressor.indexer.scorer` — 값 64 를 두고 후보가 3개, 1008축
- `c_q vs d_g` in `model.layers.*.self_attn.o_a_proj` — 값 1024 를 두고 후보가 2개, 860축
- `d_rope vs n_h` in `model.layers.*.self_attn.compressor.indexer` — 값 64 를 두고 후보가 2개, 840축
- `c_I vs w_local` in `model.layers.*.self_attn.compressor.indexer.scorer` — 값 128 를 두고 후보가 2개, 756축
- `d_rope vs n_h vs n_h_I` in `model.layers.*.self_attn.compressor.indexer.scorer.weights_proj` — 값 64 를 두고 후보가 3개, 338축
- `m_hca vs w_local` in `model.layers.*.self_attn.compressor` — 값 128 를 두고 후보가 2개, 320축
- `c_I vs w_local` in `model.layers.*.self_attn.compressor.indexer.kv_norm` — 값 128 를 두고 후보가 2개, 253축

### 0. 규칙이 끝내지 못한 축 — **여기부터 답한다**

값으로는 결정할 수 없어 파이프라인이 판단을 넘긴 자리다. 세 가지뿐이다:
`tie`(두 심볼이 같은 값이라 관례로 골랐다) · `heur`(등록 규칙이 없어 산술로 지어냈다) · `bare`(이름을 못 붙였는데 크기가 커서 진짜 차원일 수 있다).

**답이 나오면 `override_stub` 을 채워 `rules/label_overrides.yaml` 에 넣는다.** `spread: class` 라 그 축이 지나는 모든 자리가 한 번에 바뀐다 — 모듈 경계에서 멈추지 않는다(그것이 예전에 교정을 막던 유일한 이유였다).

**값이 같은 심볼이 여럿이면 값으로는 영원히 못 가른다. shape 안의 위치가 말해 준다** — `[B, n_h, T, d_head]` 의 축 1 은 head 개수, 축 3 은 head 폭이다.

아래 `shape` 과 `축` 은 **그 축을 처음 만든 자리(앵커)** 의 것이고, 초안의 `shape`/`axis` 가 그대로 그 앵커를 지목한다 — 같은 모듈에 같은 이름·같은 크기의 축이 여러 등가류로 나뉘어 있어도 서로를 훼손하지 않는다. 초안마다 유일성을 검증했다(`stub_ambiguous` 가 붙어 있으면 그 초안은 쓰지 말 것).

| 왜 | 모듈 | 크기 | 지금 이름 | 후보 | 축 | 앵커 shape | 축 수 |
|---|---|---|---|---|---|---|---|
| `tie` | `model.layers.*.self_attn` | 64 | `n_h` | `d_rope`, `n_h` | 2 | `[B, T, n_h]` | 1806 |
| `tie` | `model.layers.*.self_attn` | 64 | `n_h` | `d_rope`, `n_h` | 2 | `[B, 1, n_h]` | 1806 |
| `tie` | `model.layers.*.self_attn` | 64 | `n_h` | `d_rope`, `n_h` | 1 | `[B, n_h, T, d_rope]` | 1548 |
| `tie` | `model.layers.*.self_attn` | 64 | `d_rope` | `d_rope`, `n_h` | 3 | `[B, n_h, T, d_rope]` | 1548 |
| `tie` | `model.layers.*.self_attn` | 64 | `n_h` | `d_rope`, `n_h` | 1 | `[B, n_h, 1, d_rope]` | 1548 |
| `tie` | `model.layers.*.self_attn` | 64 | `d_rope` | `d_rope`, `n_h` | 3 | `[B, n_h, 1, d_rope]` | 1548 |
| `tie` | `model.layers.*.self_attn` | 64 | `n_h` | `d_rope`, `n_h` | 1 | `[B, n_h, T, d_head]` | 903 |
| `tie` | `model.layers.*.self_attn` | 64 | `n_h` | `d_rope`, `n_h` | 1 | `[B, n_h, 1, d_head]` | 903 |
| `tie` | `model.layers.*.self_attn.compressor` | 64 | `n_h` | `d_rope`, `n_h` | 2 | `[B, T/m_csa, n_h]` | 546 |
| `tie` | `model.layers.*.self_attn.compressor.indexer` | 64 | `n_h_I` | `d_rope`, `n_h`, `n_h_I` | 2 | `[B, T/m_csa, n_h_I]` | 525 |
| `tie` | `model.layers.*.self_attn.compressor` | 64 | `n_h` | `d_rope`, `n_h` | 2 | `[B, T/m_hca, n_h]` | 520 |
| `tie` | `model.layers.*.self_attn` | 64 | `n_h` | `d_rope`, `n_h` | 1 | `[B, n_h, T, d_rope/2]` | 516 |
| `tie` | `model.layers.*.self_attn` | 64 | `n_h` | `d_rope`, `n_h` | 1 | `[B, n_h, 1, d_rope/2]` | 516 |
| `tie` | `model.layers.*.self_attn` | 64 | `n_h` | `d_rope`, `n_h` | 2 | `[B, T, n_h, d_head]` | 430 |
| `tie` | `model.layers.*.self_attn.compressor.indexer` | 64 | `n_h_I` | `d_rope`, `n_h`, `n_h_I` | 2 | `[B, T, n_h_I, c_I]` | 420 |
| `tie` | `model.layers.*.self_attn.compressor.indexer` | 64 | `n_h_I` | `d_rope`, `n_h`, `n_h_I` | 1 | `[B, n_h_I, T, n_h]` | 420 |
| `tie` | `model.layers.*.self_attn.compressor.indexer` | 64 | `n_h` | `d_rope`, `n_h`, `n_h_I` | 3 | `[B, n_h_I, T, n_h]` | 420 |
| `tie` | `model.layers.*.self_attn.compressor.indexer` | 64 | `n_h_I` | `d_rope`, `n_h`, `n_h_I` | 2 | `[B, 1, n_h_I, c_I]` | 420 |
| `tie` | `model.layers.*.self_attn.compressor.indexer` | 128 | `c_I` | `c_I`, `w_local` | 3 | `[B, 1, n_h_I, c_I]` | 420 |
| `tie` | `model.layers.*.self_attn.compressor.indexer` | 64 | `n_h_I` | `d_rope`, `n_h`, `n_h_I` | 1 | `[B, n_h_I, 1, n_h]` | 420 |
| `tie` | `model.layers.*.self_attn.compressor.indexer` | 64 | `n_h` | `d_rope`, `n_h`, `n_h_I` | 3 | `[B, n_h_I, 1, n_h]` | 420 |
| `tie` | `model.layers.*.self_attn.q_b_norm` | 64 | `n_h` | `d_rope`, `n_h` | 1 | `[B, n_h, T, 1]` | 344 |
| `tie` | `model.layers.*.self_attn.q_b_norm` | 64 | `n_h` | `d_rope`, `n_h` | 1 | `[B, n_h, 1, 1]` | 344 |
| `tie` | `model.layers.*.self_attn.compressor.indexer` | 128 | `c_I` | `c_I`, `w_local` | 3 | `[B, T/m_csa, 2*m_csa, c_I]` | 336 |
| `tie` | `model.layers.*.self_attn.compressor.indexer` | 128 | `c_I` | `c_I`, `w_local` | 2 | `[B, T/m_csa, c_I]` | 336 |
| `tie` | `model.layers.*.self_attn` | 64 | `n_h` | `d_rope`, `n_h` | 1 | `[B, n_h, 1, 1]` | 301 |
| `tie` | `model.layers.*.self_attn.compressor.indexer` | 128 | `c_I` | `c_I`, `w_local` | 3 | `[B, T, n_h_I, c_I]` | 294 |
| `tie` | `model.layers.*.self_attn.compressor` | 128 | `m_hca` | `m_hca`, `w_local` | 2 | `[B, T/m_hca, m_hca, d_head]` | 280 |
| `tie` | `model.layers.*.self_attn` | 1024 | `d_g` | `c_q`, `d_g` | 2 | `[T, g_o, d_g]` | 258 |
| `tie` | `model.layers.*.self_attn` | 64 | `n_h` | `d_rope`, `n_h` | 2 | `[B, 1, n_h, d_head]` | 258 |
| `tie` | `model.layers.*.self_attn` | 64 | `n_h` | `d_rope`, `n_h` | 2 | `[B, 1, n_h, T+T/m_csa, d_head]` | 252 |
| `tie` | `model.layers.*.self_attn` | 64 | `n_h` | `d_rope`, `n_h` | 2 | `[B, 1, n_h, w_local+T/m_csa, d_head]` | 252 |
| `tie` | `model.layers.*.self_attn` | 64 | `n_h` | `d_rope`, `n_h` | 2 | `[B, 1, n_h, T+T/m_hca, d_head]` | 240 |
| `tie` | `model.layers.*.self_attn` | 64 | `n_h` | `d_rope`, `n_h` | 2 | `[B, 1, n_h, w_local+T/m_hca, d_head]` | 240 |
| `tie` | `model.layers.*.self_attn` | 64 | `n_h` | `d_rope`, `n_h` | 1 | `[B, n_h, T, 1]` | 215 |
| `tie` | `model.layers.*.self_attn.compressor.indexer` | 128 | `c_I` | `c_I`, `w_local` | 3 | `[B, T/m_csa, m_csa, c_I]` | 210 |
| `tie` | `model.layers.*.self_attn.compressor.indexer` | 128 | `c_I` | `c_I`, `w_local` | 3 | `[B, T/m_csa-1, m_csa, c_I]` | 210 |
| `tie` | `model.layers.*.self_attn.compressor.indexer.scorer` | 64 | `n_h_I` | `d_rope`, `n_h`, `n_h_I` | 1 | `[d_model, n_h_I]` | 210 |
| `tie` | `model.layers.*.self_attn` | 64 | `n_h` | `d_rope`, `n_h` | 1 | `[B, n_h, T, d_head-d_rope]` | 172 |
| `tie` | `model.layers.*.self_attn` | 64 | `n_h` | `d_rope`, `n_h` | 1 | `[B, n_h, T, d_rope/2, 2]` | 172 |

초안(그대로 복사해 `to` 와 `source` 만 채운다):

```yaml
  - model: deepseek-ai__DeepSeek-V4-Flash
    module: 'self_attn$'
    spread: class
    shape: ["B", "T", "n_h"]
    axis: 2
    from: n_h
    to: <소스가 말하는 이름>
    expect: 64
    source: <modeling_*.py:줄 인용>
  - model: deepseek-ai__DeepSeek-V4-Flash
    module: 'self_attn$'
    spread: class
    shape: ["B", "1", "n_h"]
    axis: 2
    from: n_h
    to: <소스가 말하는 이름>
    expect: 64
    source: <modeling_*.py:줄 인용>
  - model: deepseek-ai__DeepSeek-V4-Flash
    module: 'self_attn$'
    spread: class
    shape: ["B", "n_h", "T", "d_rope"]
    axis: 1
    from: n_h
    to: <소스가 말하는 이름>
    expect: 64
    source: <modeling_*.py:줄 인용>
  - model: deepseek-ai__DeepSeek-V4-Flash
    module: 'self_attn$'
    spread: class
    shape: ["B", "n_h", "T", "d_rope"]
    axis: 3
    from: d_rope
    to: <소스가 말하는 이름>
    expect: 64
    source: <modeling_*.py:줄 인용>
  - model: deepseek-ai__DeepSeek-V4-Flash
    module: 'self_attn$'
    spread: class
    shape: ["B", "n_h", "1", "d_rope"]
    axis: 1
    from: n_h
    to: <소스가 말하는 이름>
    expect: 64
    source: <modeling_*.py:줄 인용>
  - model: deepseek-ai__DeepSeek-V4-Flash
    module: 'self_attn$'
    spread: class
    shape: ["B", "n_h", "1", "d_rope"]
    axis: 3
    from: d_rope
    to: <소스가 말하는 이름>
    expect: 64
    source: <modeling_*.py:줄 인용>
```

## 기계적으로 이미 확인된 것 — 다시 묻지 말 것

- **심볼이 읽은 config 필드**: 전부 이 모델의 config 클래스(또는 상속/프로퍼티/getattr 기본값)에 존재한다
- **정사각 축**: `n_hc` ← 소스의 `hc` ← `hc_mult`
- **모듈이 읽는 config 속성**: `__init__` 에서 config 를 읽는 클래스 18개를 소스에서 확인했다. 그 목록이 각 모듈의 폭이 가질 수 있는 이름의 전부다.
- **가중치 축 ↔ 모듈 소속**: 가중치 축의 이름이 전부 그 모듈(또는 그 부모)이 실제로 읽는 config 필드에서 나왔다. 이 축들은 값이 아니라 소스로 확인된 것이다.

## 행 단위 전건 — 여기부터 읽는다

접힌 표(`<phase>.jsonl`)의 **고유 행 전부**다. 레이어 인덱스만 접었고 그 밖에는 아무것도 합치지 않았다. 아래 A/B/C 절은 (모듈, 라벨)로 접은 뷰라 **한 행 안의 어긋남이 보이지 않는다** — 실제로 외부 검토가 찾아낸 결함 세 건이 전부 그 자리에 있었다.

**한 행씩 읽고 이것만 물어라: 입력·가중치·출력이 서로 말이 되는가.**

- 같은 텐서가 `input_shape` 와 `weight_shape` 에서 다른 이름을 쓰지 않는가 (가중치는 `[out, in]` 으로 저장되고 피연산자는 전치돼 있다)
- 전치·view 처럼 **이름을 바꿀 수 없는 op** 이 이름을 바꾸지 않았는가
- 행렬곱의 수축 축이 양쪽에서 같은 이름인가 — `[m,k] @ [k,n] -> [m,n]`
- 이 모듈이 그 이름을 가질 수 있는가 (소스에서 그 `nn.Linear` 를 만드는 줄을 찾아라)

고유 행 136개.

| phase | 모듈 | op | input_shape | weight_shape | output_shape |
|---|---|---|---|---|---|
| prefill | `model.embed_tokens` | embedding | `[['V', 'd_model'], ['B', 'T']]` | `['V', 'd_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.attn_hc.input_norm` | rmsnorm | `[['B', 'T', 'n_hc*d_model']]` | `None` | `[['B', 'T', 'n_hc*d_model']]` |
| prefill | `model.layers.*.attn_hc` | matmul | `[['T', 'n_hc*d_model'], ['n_hc*d_model', '(2+n_hc)*n_hc']]` | `['(2+n_hc)*n_hc', 'n_hc*d_model']` | `[['T', '(2+n_hc)*n_hc']]` |
| prefill | `model.layers.*.attn_hc` | sigmoid | `[['B', 'T', 'n_hc']]` | `None` | `[['B', 'T', 'n_hc']]` |
| prefill | `model.layers.*.attn_hc` | softmax | `[['B', 'T', 'n_hc', 'n_hc']]` | `None` | `[['B', 'T', 'n_hc', 'n_hc']]` |
| prefill | `model.layers.*.attn_hc` | elementwise_mul | `[['B', 'T', 'n_hc', '1'], ['B', 'T', 'n_hc', 'd_model']]` | `None` | `[['B', 'T', 'n_hc', 'd_model']]` |
| prefill | `model.layers.*.attn_hc` | sum | `[['B', 'T', 'n_hc', 'd_model']]` | `None` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.input_layernorm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.self_attn.q_a_proj` | matmul | `[['T', 'd_model'], ['d_model', 'c_q']]` | `['c_q', 'd_model']` | `[['T', 'c_q']]` |
| prefill | `model.layers.*.self_attn.q_a_norm` | rmsnorm | `[['B', 'T', 'c_q']]` | `['c_q']` | `[['B', 'T', 'c_q']]` |
| prefill | `model.layers.*.self_attn.q_b_proj` | matmul | `[['T', 'c_q'], ['c_q', 'n_h*d_head']]` | `['n_h*d_head', 'c_q']` | `[['T', 'n_h*d_head']]` |
| prefill | `model.layers.*.self_attn.q_b_norm` | rmsnorm | `[['B', 'n_h', 'T', 'd_head']]` | `None` | `[['B', 'n_h', 'T', 'd_head']]` |
| prefill | `model.layers.*.self_attn.kv_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_head']]` | `['d_head', 'd_model']` | `[['T', 'd_head']]` |
| prefill | `model.layers.*.self_attn.kv_norm` | rmsnorm | `[['B', 'T', 'd_head']]` | `['d_head']` | `[['B', 'T', 'd_head']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'T', 'd_head'], ['n_h', 'd_head', 'T']]` | `None` | `[['n_h', 'T', 'T']]` |
| prefill | `model.layers.*.self_attn` | softmax | `[['B', 'n_h', 'T', 'T+1']]` | `None` | `[['B', 'n_h', 'T', 'T+1']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'T', 'T'], ['n_h', 'T', 'd_head']]` | `None` | `[['n_h', 'T', 'd_head']]` |
| prefill | `model.layers.*.self_attn.o_a_proj` | batched_matmul | `[['g_o', 'T', 'd_model'], ['g_o', 'd_model', 'd_g']]` | `['g_o*d_g', 'd_model']` | `[['g_o', 'T', 'd_g']]` |
| prefill | `model.layers.*.self_attn.o_b_proj` | matmul | `[['T', 'g_o*d_g'], ['g_o*d_g', 'd_model']]` | `['d_model', 'g_o*d_g']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_mul | `[['B', 'T', 'n_hc', '1'], ['B', 'T', '1', 'd_model']]` | `None` | `[['B', 'T', 'n_hc', 'd_model']]` |
| prefill | `model.layers.*` | batched_matmul | `[['T', 'n_hc', 'n_hc'], ['T', 'n_hc', 'd_model']]` | `None` | `[['T', 'n_hc', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_add | `[['B', 'T', 'n_hc', 'd_model'], ['B', 'T', 'n_hc', 'd_model']]` | `None` | `[['B', 'T', 'n_hc', 'd_model']]` |
| prefill | `model.layers.*.ffn_hc.input_norm` | rmsnorm | `[['B', 'T', 'n_hc*d_model']]` | `None` | `[['B', 'T', 'n_hc*d_model']]` |
| prefill | `model.layers.*.ffn_hc` | matmul | `[['T', 'n_hc*d_model'], ['n_hc*d_model', '(2+n_hc)*n_hc']]` | `['(2+n_hc)*n_hc', 'n_hc*d_model']` | `[['T', '(2+n_hc)*n_hc']]` |
| prefill | `model.layers.*.ffn_hc` | sigmoid | `[['B', 'T', 'n_hc']]` | `None` | `[['B', 'T', 'n_hc']]` |
| prefill | `model.layers.*.ffn_hc` | softmax | `[['B', 'T', 'n_hc', 'n_hc']]` | `None` | `[['B', 'T', 'n_hc', 'n_hc']]` |
| prefill | `model.layers.*.ffn_hc` | elementwise_mul | `[['B', 'T', 'n_hc', '1'], ['B', 'T', 'n_hc', 'd_model']]` | `None` | `[['B', 'T', 'n_hc', 'd_model']]` |
| prefill | `model.layers.*.ffn_hc` | sum | `[['B', 'T', 'n_hc', 'd_model']]` | `None` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.post_attention_layernorm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.mlp.gate` | matmul | `[['T', 'd_model'], ['d_model', 'E']]` | `['E', 'd_model']` | `[['T', 'E']]` |
| prefill | `model.layers.*.mlp.experts` | grouped_matmul | `[['k*T', 'd_model'], ['E', 'd_model', 'd_model'], ['E']]` | `['E', 'd_model', 'd_model']` | `[['k*T', '2*d_moe']]` |
| prefill | `model.layers.*.mlp.experts.act_fn` | silu | `[['k*T', 'd_moe']]` | `None` | `[['k*T', 'd_moe']]` |
| prefill | `model.layers.*.mlp.experts` | elementwise_mul | `[['k*T', 'd_moe'], ['k*T', 'd_moe']]` | `None` | `[['k*T', 'd_moe']]` |
| prefill | `model.layers.*.mlp.experts` | grouped_matmul | `[['k*T', 'd_moe'], ['E', 'd_moe', 'd_model'], ['E']]` | `['E', 'd_model', 'd_moe']` | `[['k*T', 'd_model']]` |
| prefill | `model.layers.*.mlp.experts` | elementwise_mul | `[['k*T', 'd_model'], ['k*T', 'B']]` | `None` | `[['k*T', 'd_model']]` |
| prefill | `model.layers.*.mlp.experts` | sum | `[['T', 'k', 'd_model']]` | `None` | `[['T', 'd_model']]` |
| prefill | `model.layers.*.mlp.shared_experts.gate_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_moe']]` | `['d_moe', 'd_model']` | `[['T', 'd_moe']]` |
| prefill | `model.layers.*.mlp.shared_experts.up_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_moe']]` | `['d_moe', 'd_model']` | `[['T', 'd_moe']]` |
| prefill | `model.layers.*.mlp.shared_experts.act_fn` | silu | `[['B', 'T', 'd_moe']]` | `None` | `[['B', 'T', 'd_moe']]` |
| prefill | `model.layers.*.mlp.shared_experts` | elementwise_mul | `[['B', 'T', 'd_moe'], ['B', 'T', 'd_moe']]` | `None` | `[['B', 'T', 'd_moe']]` |
| prefill | `model.layers.*.mlp.shared_experts.down_proj` | matmul | `[['T', 'd_moe'], ['d_moe', 'd_model']]` | `['d_model', 'd_moe']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*.mlp` | elementwise_add | `[['B', 'T', 'd_model'], ['B', 'T', 'd_model']]` | `None` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.self_attn.compressor.kv_proj` | matmul | `[['T', 'd_model'], ['d_model', 'c_q']]` | `['c_q', 'd_model']` | `[['T', 'c_q']]` |
| prefill | `model.layers.*.self_attn.compressor.gate_proj` | matmul | `[['T', 'd_model'], ['d_model', 'c_q']]` | `['c_q', 'd_model']` | `[['T', 'c_q']]` |
| prefill | `model.layers.*.self_attn.compressor` | softmax | `[['B', 'T/m_csa', 'T/m_hca', 'd_head']]` | `None` | `[['B', 'T/m_csa', 'T/m_hca', 'd_head']]` |
| prefill | `model.layers.*.self_attn.compressor.kv_norm` | rmsnorm | `[['B', 'T/m_csa', 'd_head']]` | `['d_head']` | `[['B', 'T/m_csa', 'd_head']]` |
| prefill | `model.layers.*.self_attn.compressor.indexer.kv_proj` | matmul | `[['T', 'd_model'], ['d_model', '2*c_I']]` | `['2*c_I', 'd_model']` | `[['T', '2*c_I']]` |
| prefill | `model.layers.*.self_attn.compressor.indexer.gate_proj` | matmul | `[['T', 'd_model'], ['d_model', '2*c_I']]` | `['2*c_I', 'd_model']` | `[['T', '2*c_I']]` |
| prefill | `model.layers.*.self_attn.compressor.indexer` | softmax | `[['B', 'T/m_csa', '2*m_csa', 'c_I']]` | `None` | `[['B', 'T/m_csa', '2*m_csa', 'c_I']]` |
| prefill | `model.layers.*.self_attn.compressor.indexer.kv_norm` | rmsnorm | `[['B', 'T/m_csa', 'c_I']]` | `['c_I']` | `[['B', 'T/m_csa', 'c_I']]` |
| prefill | `model.layers.*.self_attn.compressor.indexer.q_b_proj` | matmul | `[['T', 'c_q'], ['c_q', 'n_h_I*c_I']]` | `['n_h_I*c_I', 'c_q']` | `[['T', 'n_h_I*c_I']]` |
| prefill | `model.layers.*.self_attn.compressor.indexer.scorer` | batched_matmul | `[['T', 'n_h_I', 'c_I'], ['T', 'c_I', 'T/m_csa']]` | `None` | `[['T', 'n_h_I', 'T/m_csa']]` |
| prefill | `model.layers.*.self_attn.compressor.indexer.scorer` | relu | `[['B', 'T', 'n_h_I', 'T/m_csa']]` | `None` | `[['B', 'T', 'n_h_I', 'T/m_csa']]` |
| prefill | `model.layers.*.self_attn.compressor.indexer.scorer.weights_proj` | matmul | `[['T', 'd_model'], ['d_model', 'n_h_I']]` | `['n_h_I', 'd_model']` | `[['T', 'n_h_I']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'T', 'd_head'], ['n_h', 'd_head', 'T+T/m_csa']]` | `None` | `[['n_h', 'T', 'T+T/m_csa']]` |
| prefill | `model.layers.*.self_attn` | softmax | `[['B', 'n_h', 'T', 'T+T/m_csa+1']]` | `None` | `[['B', 'n_h', 'T', 'T+T/m_csa+1']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'T', 'T+T/m_csa'], ['n_h', 'T+T/m_csa', 'd_head']]` | `None` | `[['n_h', 'T', 'd_head']]` |
| prefill | `model.layers.*.self_attn.compressor.kv_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_head']]` | `['d_head', 'd_model']` | `[['T', 'd_head']]` |
| prefill | `model.layers.*.self_attn.compressor.gate_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_head']]` | `['d_head', 'd_model']` | `[['T', 'd_head']]` |
| prefill | `model.layers.*.self_attn.compressor` | softmax | `[['B', 'T/m_hca', 'm_hca', 'd_head']]` | `None` | `[['B', 'T/m_hca', 'm_hca', 'd_head']]` |
| prefill | `model.layers.*.self_attn.compressor.kv_norm` | rmsnorm | `[['B', 'T/m_hca', 'd_head']]` | `['d_head']` | `[['B', 'T/m_hca', 'd_head']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'T', 'd_head'], ['n_h', 'd_head', 'T+T/m_hca']]` | `None` | `[['n_h', 'T', 'T+T/m_hca']]` |
| prefill | `model.layers.*.self_attn` | softmax | `[['B', 'n_h', 'T', 'T+T/m_hca+1']]` | `None` | `[['B', 'n_h', 'T', 'T+T/m_hca+1']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'T', 'T+T/m_hca'], ['n_h', 'T+T/m_hca', 'd_head']]` | `None` | `[['n_h', 'T', 'd_head']]` |
| prefill | `model.hc_head.input_norm` | rmsnorm | `[['B', 'T', 'n_hc*d_model']]` | `None` | `[['B', 'T', 'n_hc*d_model']]` |
| prefill | `model.hc_head` | matmul | `[['T', 'n_hc*d_model'], ['n_hc*d_model', 'n_hc']]` | `['n_hc', 'n_hc*d_model']` | `[['T', 'n_hc']]` |
| prefill | `model.hc_head` | sigmoid | `[['B', 'T', 'n_hc']]` | `None` | `[['B', 'T', 'n_hc']]` |
| prefill | `model.hc_head` | elementwise_mul | `[['B', 'T', 'n_hc', '1'], ['B', 'T', 'n_hc', 'd_model']]` | `None` | `[['B', 'T', 'n_hc', 'd_model']]` |
| prefill | `model.hc_head` | sum | `[['B', 'T', 'n_hc', 'd_model']]` | `None` | `[['B', 'T', 'd_model']]` |
| prefill | `model.norm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `lm_head` | matmul | `[['T', 'd_model'], ['d_model', 'V']]` | `['V', 'd_model']` | `[['T', 'V']]` |
| decode | `model.embed_tokens` | embedding | `[['V', 'd_model'], ['B', '1']]` | `['V', 'd_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.attn_hc.input_norm` | rmsnorm | `[['B', '1', 'n_hc*d_model']]` | `None` | `[['B', '1', 'n_hc*d_model']]` |
| decode | `model.layers.*.attn_hc` | matmul | `[['B', 'n_hc*d_model'], ['n_hc*d_model', '(2+n_hc)*n_hc']]` | `['(2+n_hc)*n_hc', 'n_hc*d_model']` | `[['B', '(2+n_hc)*n_hc']]` |
| decode | `model.layers.*.attn_hc` | sigmoid | `[['B', '1', 'n_hc']]` | `None` | `[['B', '1', 'n_hc']]` |
| decode | `model.layers.*.attn_hc` | softmax | `[['B', '1', 'n_hc', 'n_hc']]` | `None` | `[['B', '1', 'n_hc', 'n_hc']]` |
| decode | `model.layers.*.attn_hc` | elementwise_mul | `[['B', '1', 'n_hc', '1'], ['B', '1', 'n_hc', 'd_model']]` | `None` | `[['B', '1', 'n_hc', 'd_model']]` |
| decode | `model.layers.*.attn_hc` | sum | `[['B', '1', 'n_hc', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.input_layernorm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.self_attn.q_a_proj` | matmul | `[['B', 'd_model'], ['d_model', 'c_q']]` | `['c_q', 'd_model']` | `[['B', 'c_q']]` |
| decode | `model.layers.*.self_attn.q_a_norm` | rmsnorm | `[['B', '1', 'c_q']]` | `['c_q']` | `[['B', '1', 'c_q']]` |
| decode | `model.layers.*.self_attn.q_b_proj` | matmul | `[['B', 'c_q'], ['c_q', 'n_h*d_head']]` | `['n_h*d_head', 'c_q']` | `[['B', 'n_h*d_head']]` |
| decode | `model.layers.*.self_attn.q_b_norm` | rmsnorm | `[['B', 'n_h', '1', 'd_head']]` | `None` | `[['B', 'n_h', '1', 'd_head']]` |
| decode | `model.layers.*.self_attn.kv_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_head']]` | `['d_head', 'd_model']` | `[['B', 'd_head']]` |
| decode | `model.layers.*.self_attn.kv_norm` | rmsnorm | `[['B', '1', 'd_head']]` | `['d_head']` | `[['B', '1', 'd_head']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'B', 'd_head'], ['n_h', 'd_head', 'w_local']]` | `None` | `[['n_h', 'B', 'w_local']]` |
| decode | `model.layers.*.self_attn` | softmax | `[['B', 'n_h', '1', 'w_local+1']]` | `None` | `[['B', 'n_h', '1', 'w_local+1']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'B', 'w_local'], ['n_h', 'w_local', 'd_head']]` | `None` | `[['n_h', 'B', 'd_head']]` |
| decode | `model.layers.*.self_attn.o_a_proj` | batched_matmul | `[['g_o', 'B', 'd_model'], ['g_o', 'd_model', 'd_g']]` | `['g_o*d_g', 'd_model']` | `[['g_o', 'B', 'd_g']]` |
| decode | `model.layers.*.self_attn.o_b_proj` | matmul | `[['B', 'g_o*d_g'], ['g_o*d_g', 'd_model']]` | `['d_model', 'g_o*d_g']` | `[['B', 'd_model']]` |
| decode | `model.layers.*` | elementwise_mul | `[['B', '1', 'n_hc', '1'], ['B', '1', '1', 'd_model']]` | `None` | `[['B', '1', 'n_hc', 'd_model']]` |
| decode | `model.layers.*` | batched_matmul | `[['B', 'n_hc', 'n_hc'], ['B', 'n_hc', 'd_model']]` | `None` | `[['B', 'n_hc', 'd_model']]` |
| decode | `model.layers.*` | elementwise_add | `[['B', '1', 'n_hc', 'd_model'], ['B', '1', 'n_hc', 'd_model']]` | `None` | `[['B', '1', 'n_hc', 'd_model']]` |
| decode | `model.layers.*.ffn_hc.input_norm` | rmsnorm | `[['B', '1', 'n_hc*d_model']]` | `None` | `[['B', '1', 'n_hc*d_model']]` |
| decode | `model.layers.*.ffn_hc` | matmul | `[['B', 'n_hc*d_model'], ['n_hc*d_model', '(2+n_hc)*n_hc']]` | `['(2+n_hc)*n_hc', 'n_hc*d_model']` | `[['B', '(2+n_hc)*n_hc']]` |
| decode | `model.layers.*.ffn_hc` | sigmoid | `[['B', '1', 'n_hc']]` | `None` | `[['B', '1', 'n_hc']]` |
| decode | `model.layers.*.ffn_hc` | softmax | `[['B', '1', 'n_hc', 'n_hc']]` | `None` | `[['B', '1', 'n_hc', 'n_hc']]` |
| decode | `model.layers.*.ffn_hc` | elementwise_mul | `[['B', '1', 'n_hc', '1'], ['B', '1', 'n_hc', 'd_model']]` | `None` | `[['B', '1', 'n_hc', 'd_model']]` |
| decode | `model.layers.*.ffn_hc` | sum | `[['B', '1', 'n_hc', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.post_attention_layernorm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.mlp.gate` | matmul | `[['B', 'd_model'], ['d_model', 'E']]` | `['E', 'd_model']` | `[['B', 'E']]` |
| decode | `model.layers.*.mlp.experts` | grouped_matmul | `[['k', 'd_model'], ['E', 'd_model', 'd_model'], ['E']]` | `['E', 'd_model', 'd_model']` | `[['k', '2*d_moe']]` |
| decode | `model.layers.*.mlp.experts.act_fn` | silu | `[['k', 'd_moe']]` | `None` | `[['k', 'd_moe']]` |
| decode | `model.layers.*.mlp.experts` | elementwise_mul | `[['k', 'd_moe'], ['k', 'd_moe']]` | `None` | `[['k', 'd_moe']]` |
| decode | `model.layers.*.mlp.experts` | grouped_matmul | `[['k', 'd_moe'], ['E', 'd_moe', 'd_model'], ['E']]` | `['E', 'd_model', 'd_moe']` | `[['k', 'd_model']]` |
| decode | `model.layers.*.mlp.experts` | elementwise_mul | `[['k', 'd_model'], ['k', 'B']]` | `None` | `[['k', 'd_model']]` |
| decode | `model.layers.*.mlp.experts` | sum | `[['B', 'k', 'd_model']]` | `None` | `[['B', 'd_model']]` |
| decode | `model.layers.*.mlp.shared_experts.gate_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_moe']]` | `['d_moe', 'd_model']` | `[['B', 'd_moe']]` |
| decode | `model.layers.*.mlp.shared_experts.up_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_moe']]` | `['d_moe', 'd_model']` | `[['B', 'd_moe']]` |
| decode | `model.layers.*.mlp.shared_experts.act_fn` | silu | `[['B', '1', 'd_moe']]` | `None` | `[['B', '1', 'd_moe']]` |
| decode | `model.layers.*.mlp.shared_experts` | elementwise_mul | `[['B', '1', 'd_moe'], ['B', '1', 'd_moe']]` | `None` | `[['B', '1', 'd_moe']]` |
| decode | `model.layers.*.mlp.shared_experts.down_proj` | matmul | `[['B', 'd_moe'], ['d_moe', 'd_model']]` | `['d_model', 'd_moe']` | `[['B', 'd_model']]` |
| decode | `model.layers.*.mlp` | elementwise_add | `[['B', '1', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.self_attn.compressor.kv_proj` | matmul | `[['B', 'd_model'], ['d_model', 'c_q']]` | `['c_q', 'd_model']` | `[['B', 'c_q']]` |
| decode | `model.layers.*.self_attn.compressor.gate_proj` | matmul | `[['B', 'd_model'], ['d_model', 'c_q']]` | `['c_q', 'd_model']` | `[['B', 'c_q']]` |
| decode | `model.layers.*.self_attn.compressor.indexer.kv_proj` | matmul | `[['B', 'd_model'], ['d_model', '2*c_I']]` | `['2*c_I', 'd_model']` | `[['B', '2*c_I']]` |
| decode | `model.layers.*.self_attn.compressor.indexer.gate_proj` | matmul | `[['B', 'd_model'], ['d_model', '2*c_I']]` | `['2*c_I', 'd_model']` | `[['B', '2*c_I']]` |
| decode | `model.layers.*.self_attn.compressor.indexer.q_b_proj` | matmul | `[['B', 'c_q'], ['c_q', 'n_h_I*c_I']]` | `['n_h_I*c_I', 'c_q']` | `[['B', 'n_h_I*c_I']]` |
| decode | `model.layers.*.self_attn.compressor.indexer.scorer` | batched_matmul | `[['B', 'n_h_I', 'c_I'], ['B', 'c_I', 'T/m_csa']]` | `None` | `[['B', 'n_h_I', 'T/m_csa']]` |
| decode | `model.layers.*.self_attn.compressor.indexer.scorer` | relu | `[['B', '1', 'n_h_I', 'T/m_csa']]` | `None` | `[['B', '1', 'n_h_I', 'T/m_csa']]` |
| decode | `model.layers.*.self_attn.compressor.indexer.scorer.weights_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_h_I']]` | `['n_h_I', 'd_model']` | `[['B', 'n_h_I']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'B', 'd_head'], ['n_h', 'd_head', 'w_local+T/m_csa']]` | `None` | `[['n_h', 'B', 'w_local+T/m_csa']]` |
| decode | `model.layers.*.self_attn` | softmax | `[['B', 'n_h', '1', 'w_local+T/m_csa+1']]` | `None` | `[['B', 'n_h', '1', 'w_local+T/m_csa+1']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'B', 'w_local+T/m_csa'], ['n_h', 'w_local+T/m_csa', 'd_head']]` | `None` | `[['n_h', 'B', 'd_head']]` |
| decode | `model.layers.*.self_attn.compressor.kv_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_head']]` | `['d_head', 'd_model']` | `[['B', 'd_head']]` |
| decode | `model.layers.*.self_attn.compressor.gate_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_head']]` | `['d_head', 'd_model']` | `[['B', 'd_head']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'B', 'd_head'], ['n_h', 'd_head', 'w_local+T/m_hca']]` | `None` | `[['n_h', 'B', 'w_local+T/m_hca']]` |
| decode | `model.layers.*.self_attn` | softmax | `[['B', 'n_h', '1', 'w_local+T/m_hca+1']]` | `None` | `[['B', 'n_h', '1', 'w_local+T/m_hca+1']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'B', 'w_local+T/m_hca'], ['n_h', 'w_local+T/m_hca', 'd_head']]` | `None` | `[['n_h', 'B', 'd_head']]` |
| decode | `model.hc_head.input_norm` | rmsnorm | `[['B', '1', 'n_hc*d_model']]` | `None` | `[['B', '1', 'n_hc*d_model']]` |
| decode | `model.hc_head` | matmul | `[['B', 'n_hc*d_model'], ['n_hc*d_model', 'n_hc']]` | `['n_hc', 'n_hc*d_model']` | `[['B', 'n_hc']]` |
| decode | `model.hc_head` | sigmoid | `[['B', '1', 'n_hc']]` | `None` | `[['B', '1', 'n_hc']]` |
| decode | `model.hc_head` | elementwise_mul | `[['B', '1', 'n_hc', '1'], ['B', '1', 'n_hc', 'd_model']]` | `None` | `[['B', '1', 'n_hc', 'd_model']]` |
| decode | `model.hc_head` | sum | `[['B', '1', 'n_hc', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.norm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `lm_head` | matmul | `[['B', 'd_model'], ['d_model', 'V']]` | `['V', 'd_model']` | `[['B', 'V']]` |

## 전수 점검 — 이 모델이 쓰는 이름 전부

위 절이 '풀리지 않은 것'이라면 여기는 **전부**다. 규칙이 자신 있게 붙인 이름도 틀릴 수 있고, 그런 건 미결 목록에 절대 오르지 않는다. 한 줄씩 읽고 **그 모듈에서 그 이름이 말이 되는지** 보라.

### A. 붙은 이름 전부 (47종)

| 라벨 | 값 | 나타나는 모듈 | 축 수 |
|---|---|---|---|
| `B` |  | `model.layers.*.attn_hc`, `model.layers.*.ffn_hc`, `model.layers.*.self_attn`, `model.layers.*.self_attn.compressor.indexer` 외 84개 | 116557 |
| `n_hc` | 4 | `model.layers.*.attn_hc`, `model.layers.*.ffn_hc`, `model.layers.0`, `model.layers.1` 외 43개 | 81936 |
| `T` |  | `model.layers.*.attn_hc`, `model.layers.*.ffn_hc`, `model.layers.*.self_attn`, `model.layers.*.self_attn.compressor.indexer` 외 81개 | 53950 |
| `n_h` | 64 | `model.layers.*.self_attn`, `model.layers.*.self_attn.q_b_norm`, `model.layers.*.self_attn.compressor`, `model.layers.*.self_attn.compressor.indexer` | 17902 |
| `d_model` | 4096 | `model.layers.*.mlp.experts`, `model.layers.*.self_attn.o_a_proj`, `model.layers.*.input_layernorm`, `model.layers.*.post_attention_layernorm` 외 65개 | 17532 |
| `d_head` | 512 | `model.layers.*.self_attn`, `model.layers.*.self_attn.compressor`, `model.layers.*.self_attn.kv_norm`, `model.layers.*.self_attn.kv_proj` 외 4개 | 11388 |
| `d_rope/2` |  | `model.layers.*.self_attn`, `model.layers.*.self_attn.compressor.indexer.rotary_emb`, `model.layers.*.self_attn.compressor.indexer`, `model.layers.*.self_attn.compressor.rotary_emb` 외 2개 | 11160 |
| `T/m_csa` |  | `model.layers.*.self_attn.compressor.indexer`, `model.layers.*.self_attn.compressor`, `model.layers.*.self_attn.compressor.indexer.scorer`, `model.layers.*.self_attn.compressor.rotary_emb` 외 4개 | 8652 |
| `c_q` | 1024 | `model.layers.*.self_attn.q_a_norm`, `model.layers.*.self_attn.compressor`, `model.layers.*.self_attn.q_a_proj`, `model.layers.*.self_attn.q_b_proj` 외 3개 | 4357 |
| `d_moe` | 2048 | `model.layers.*.mlp.experts`, `model.layers.*.mlp.shared_experts.gate_proj`, `model.layers.*.mlp.shared_experts.up_proj`, `model.layers.*.mlp.shared_experts.down_proj` 외 3개 | 4300 |
| `n_h_I` | 64 | `model.layers.*.self_attn.compressor.indexer`, `model.layers.*.self_attn.compressor.indexer.scorer`, `model.layers.*.self_attn.compressor.indexer.scorer.weights_proj` | 3780 |
| `T/m_hca` |  | `model.layers.*.self_attn.compressor`, `model.layers.*.self_attn.compressor.rotary_emb`, `model.layers.*.self_attn.compressor.kv_norm`, `model.layers.*.self_attn.o_a_proj` 외 1개 | 3664 |
| `k` | 6 | `model.layers.*.mlp.experts`, `model.layers.*.mlp.gate`, `model.layers.*.mlp.experts.act_fn` | 3661 |
| `n_hc*d_model` |  | `model.layers.*.attn_hc`, `model.layers.*.ffn_hc`, `model.layers.*.attn_hc.input_norm`, `model.layers.*.ffn_hc.input_norm` 외 2개 | 3306 |
| `d_rope` | 64 | `model.layers.*.self_attn` | 3096 |
| `E` | 256 | `model.layers.*.mlp.experts`, `model.layers.*.mlp.gate`, `model.layers.*.mlp.gate.score_fn` | 2556 |
| `k*T` |  | `model.layers.*.mlp.experts`, `model.layers.*.mlp.experts.act_fn` | 2537 |
| `c_I` | 128 | `model.layers.*.self_attn.compressor.indexer`, `model.layers.*.self_attn.compressor.indexer.scorer`, `model.layers.*.self_attn.compressor.indexer.kv_norm` | 2499 |
| `(2+n_hc)*n_hc` |  | `model.layers.*.attn_hc`, `model.layers.*.ffn_hc` | 2236 |
| `m_csa` | 4 | `model.layers.*.self_attn.compressor`, `model.layers.*.self_attn.compressor.indexer` | 2016 |
| `2*c_I` |  | `model.layers.*.self_attn.compressor.indexer`, `model.layers.*.self_attn.compressor.indexer.kv_proj`, `model.layers.*.self_attn.compressor.indexer.gate_proj` | 1533 |
| `g_o*d_g` |  | `model.layers.*.self_attn.o_b_proj`, `model.layers.*.self_attn.o_a_proj`, `model.layers.*.self_attn` | 1118 |
| `g_o` | 8 | `model.layers.*.self_attn.o_a_proj`, `model.layers.*.self_attn` | 1118 |
| `d_g` | 1024 | `model.layers.*.self_attn.o_a_proj`, `model.layers.*.self_attn` | 946 |
| `T+T/m_csa` |  | `model.layers.*.self_attn` | 861 |
| `w_local+T/m_csa` |  | `model.layers.*.self_attn` | 861 |
| `T+T/m_hca` |  | `model.layers.*.self_attn` | 820 |
| `w_local+T/m_hca` |  | `model.layers.*.self_attn` | 820 |
| `n_h*d_head` |  | `model.layers.*.self_attn.q_b_proj`, `model.layers.*.self_attn` | 774 |
| `T/m_csa-1` |  | `model.layers.*.self_attn.compressor`, `model.layers.*.self_attn.compressor.indexer` | 756 |
| `n_hc*n_hc` |  | `model.layers.*.attn_hc`, `model.layers.*.ffn_hc` | 688 |
| `d_head-d_rope` |  | `model.layers.*.self_attn`, `model.layers.*.self_attn.compressor` | 598 |
| `2*m_csa` |  | `model.layers.*.self_attn.compressor.indexer` | 420 |
| `n_h_I*c_I` |  | `model.layers.*.self_attn.compressor.indexer.q_b_proj`, `model.layers.*.self_attn.compressor.indexer` | 378 |
| `m_hca` | 128 | `model.layers.*.self_attn.compressor` | 320 |
| `w_local` | 128 | `model.layers.*.self_attn`, `model` | 272 |
| `2*d_moe` |  | `model.layers.*.mlp.experts` | 172 |
| `T/m_csa+1` |  | `model.layers.*.self_attn.compressor` | 168 |
| `T/m_hca+1` |  | `model.layers.*.self_attn.compressor` | 160 |
| `T+T/m_csa+1` |  | `model.layers.*.self_attn` | 147 |
| `w_local+T/m_csa+1` |  | `model.layers.*.self_attn` | 147 |
| `T+T/m_hca+1` |  | `model.layers.*.self_attn` | 140 |
| `w_local+T/m_hca+1` |  | `model.layers.*.self_attn` | 140 |
| `w_local-1` |  | `model.layers.*.self_attn` | 135 |
| `V` | 129280 | `lm_head`, `model.layers.*.mlp.gate`, `model.embed_tokens` | 32 |
| `T+1` |  | `model.layers.*.self_attn` | 14 |
| `w_local+1` |  | `model.layers.*.self_attn` | 14 |

### B. 이름 없이 남은 정수 전부 (5쌍)

**여기가 필터가 못 보던 자리다.** 정수가 남는 것 자체는 정상이다(루프 인덱스, 피연산자 개수, 브로드캐스트 축). 문제는 **이름이 있어야 하는데 없는 경우**이고, 마지막 열이 그 신호다 — 이 모델의 심볼과 값이 같다면 스코프가 그 모듈을 못 덮고 있을 수 있다. 실제로 `n_hc`(=4)가 그렇게 정수로 남아 있었다.

| 모듈 | 정수 | 축 수 | 같은 값의 심볼 |
|---|---|---|---|
| `model.layers.*.self_attn` | 2 | 2580 | — |
| `model.layers.*.self_attn.compressor.indexer` | 2 | 630 | — |
| `model.layers.*.self_attn.compressor` | 2 | 410 | — |
| `model.layers.*.attn_hc` | 3 | 86 | — |
| `model.layers.*.ffn_hc` | 3 | 86 | — |

### C. 모듈이 내는 출력 shape 전부 (89개 모듈 / 1149종)

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
  - `[[B, 1, 1, 1]]`
  - `[[B, 1, 1, T]]`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, 1, w_local]]`
  - `[[B, 1, 1]]`
  - `[[B, 1, T, 1]]`
  - `[[B, 1, T, T]]`
  - `[[B, 1, T]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, w_local]]`
  - `[[B, 1]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T]]`
  - `[[B, w_local]]`
  - `[[B]]`
  - `[[T]]`
  - `[[]]`
  - `[[w_local]]`
- `model.embed_tokens`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.hc_head`
  - `[[B, 1, d_model]]`
  - `[[B, 1, n_hc*d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, d_model]]`
  - `[[B, T, n_hc*d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc*d_model]]`
  - `[[B, n_hc]]`
  - `[[B]]`
  - `[[T, n_hc*d_model]]`
  - `[[T, n_hc]]`
  - `[[n_hc*d_model, n_hc]]`
  - `[[n_hc, n_hc*d_model]]`
  - `[[n_hc]]`
- `model.hc_head.input_norm`
  - `[[B, 1, 1]]`
  - `[[B, 1, n_hc*d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, n_hc*d_model]]`
- `model.layers.*.attn_hc`
  - `[[(2+n_hc)*n_hc, n_hc*d_model]]`
  - `[[B, (2+n_hc)*n_hc]]`
  - `[[B, 1, (2+n_hc)*n_hc]]`
  - `[[B, 1, 1, n_hc]]`
  - `[[B, 1, d_model]]`
  - `[[B, 1, n_hc*d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc], [B, 1, n_hc], [B, 1, n_hc*n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, (2+n_hc)*n_hc]]`
  - `[[B, T, 1, n_hc]]`
  - `[[B, T, d_model]]`
  - `[[B, T, n_hc*d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc], [B, T, n_hc], [B, T, n_hc*n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc*d_model]]`
  - `[[T, (2+n_hc)*n_hc]]`
  - `[[T, n_hc*d_model]]`
  - `[[], [], []]`
  - `[[n_hc*d_model, (2+n_hc)*n_hc]]`
  - `[[n_hc, n_hc]]`
  - `[[n_hc], [n_hc], [n_hc*n_hc]]`
- `model.layers.*.attn_hc.input_norm`
  - `[[B, 1, 1]]`
  - `[[B, 1, n_hc*d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, n_hc*d_model]]`
- `model.layers.*.ffn_hc`
  - `[[(2+n_hc)*n_hc, n_hc*d_model]]`
  - `[[B, (2+n_hc)*n_hc]]`
  - `[[B, 1, (2+n_hc)*n_hc]]`
  - `[[B, 1, 1, n_hc]]`
  - `[[B, 1, d_model]]`
  - `[[B, 1, n_hc*d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc], [B, 1, n_hc], [B, 1, n_hc*n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, (2+n_hc)*n_hc]]`
  - `[[B, T, 1, n_hc]]`
  - `[[B, T, d_model]]`
  - `[[B, T, n_hc*d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc], [B, T, n_hc], [B, T, n_hc*n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc*d_model]]`
  - `[[T, (2+n_hc)*n_hc]]`
  - `[[T, n_hc*d_model]]`
  - `[[], [], []]`
  - `[[n_hc*d_model, (2+n_hc)*n_hc]]`
  - `[[n_hc, n_hc]]`
  - `[[n_hc], [n_hc], [n_hc*n_hc]]`
- `model.layers.*.ffn_hc.input_norm`
  - `[[B, 1, 1]]`
  - `[[B, 1, n_hc*d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, n_hc*d_model]]`
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
  - `[[B, 1]]`
  - `[[B, E]]`
  - `[[B, d_model]]`
  - `[[B, k], [B, k]]`
  - `[[B, k]]`
  - `[[B]]`
  - `[[T, 1]]`
  - `[[T, E]]`
  - `[[T, d_model]]`
  - `[[T, k], [T, k]]`
  - `[[T, k]]`
  - `[[T]]`
  - `[[d_model, E]]`
- `model.layers.*.mlp.gate.score_fn`
  - `[[B, E]]`
  - `[[T, E]]`
- `model.layers.*.mlp.shared_experts`
  - `[[B, 1, d_moe]]`
  - `[[B, T, d_moe]]`
- `model.layers.*.mlp.shared_experts.act_fn`
  - `[[B, 1, d_moe]]`
  - `[[B, T, d_moe]]`
- `model.layers.*.mlp.shared_experts.down_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[B, d_moe]]`
  - `[[T, d_model]]`
  - `[[T, d_moe]]`
  - `[[d_moe, d_model]]`
- `model.layers.*.mlp.shared_experts.gate_proj`
  - `[[B, 1, d_moe]]`
  - `[[B, T, d_moe]]`
  - `[[B, d_model]]`
  - `[[B, d_moe]]`
  - `[[T, d_model]]`
  - `[[T, d_moe]]`
  - `[[d_model, d_moe]]`
- `model.layers.*.mlp.shared_experts.up_proj`
  - `[[B, 1, d_moe]]`
  - `[[B, T, d_moe]]`
  - `[[B, d_model]]`
  - `[[B, d_moe]]`
  - `[[T, d_model]]`
  - `[[T, d_moe]]`
  - `[[d_model, d_moe]]`
- `model.layers.*.post_attention_layernorm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
- `model.layers.*.self_attn`
  - `[[B, 1, 1, T+T/m_csa, d_head]]`
  - `[[B, 1, 1, T+T/m_hca, d_head]]`
  - `[[B, 1, 1, T, d_head]]`
  - `[[B, 1, 1, d_head-d_rope]]`
  - `[[B, 1, 1, d_head]]`
  - `[[B, 1, 1, d_rope/2, 2]]`
  - `[[B, 1, 1, d_rope/2]]`
  - `[[B, 1, 1, n_h]]`
  - `[[B, 1, 1, w_local+T/m_csa, d_head]]`
  - `[[B, 1, 1, w_local+T/m_csa]]`
  - `[[B, 1, 1, w_local+T/m_hca, d_head]]`
  - `[[B, 1, 1, w_local+T/m_hca]]`
  - `[[B, 1, 1, w_local, d_head]]`
  - `[[B, 1, T+T/m_csa, d_head]]`
  - `[[B, 1, T+T/m_hca, d_head]]`
  - `[[B, 1, T, T+T/m_csa]]`
  - `[[B, 1, T, T+T/m_hca]]`
  - `[[B, 1, T, d_head-d_rope]]`
  - `[[B, 1, T, d_head]]`
  - `[[B, 1, T, d_rope/2, 2]]`
  - `[[B, 1, T, d_rope/2]]`
  - `[[B, 1, T, n_h]]`
  - `[[B, 1, T/m_hca, d_model]]`
  - `[[B, 1, d_rope/2, 1]]`
  - `[[B, 1, d_rope/2, 2]]`
  - `[[B, 1, d_rope/2]]`
  - `[[B, 1, g_o*d_g]]`
  - `[[B, 1, n_h, T+T/m_csa, d_head]]`
  - `[[B, 1, n_h, T+T/m_hca, d_head]]`
  - `[[B, 1, n_h, T, d_head]]`
  - `[[B, 1, n_h, d_head]]`
  - `[[B, 1, n_h, w_local+T/m_csa, d_head]]`
  - `[[B, 1, n_h, w_local+T/m_hca, d_head]]`
  - `[[B, 1, n_h, w_local, d_head]]`
  - `[[B, 1, n_h]]`
  - `[[B, 1, w_local+T/m_csa, d_head]]`
  - `[[B, 1, w_local+T/m_hca, d_head]]`
  - `[[B, 1, w_local, d_head]]`
  - `[[B, 1, w_local-1, d_head]]`
  - `[[B, T, 1, d_head]]`
  - `[[B, T, T/m_hca, d_model]]`
  - `[[B, T, d_rope/2, 1]]`
  - `[[B, T, d_rope/2, 2]]`
  - `[[B, T, d_rope/2]]`
  - `[[B, T, g_o*d_g]]`
  - `[[B, T, g_o, d_g]]`
  - `[[B, T, n_h, d_head]]`
  - `[[B, T, n_h]]`
  - `[[B, n_h, 1, 1], [B, n_h, 1, 1]]`
  - `[[B, n_h, 1, 1]]`
  - `[[B, n_h, 1, d_head-d_rope]]`
  - `[[B, n_h, 1, d_head]]`
  - `[[B, n_h, 1, d_rope/2, 2]]`
  - `[[B, n_h, 1, d_rope/2]]`
  - `[[B, n_h, 1, d_rope]]`
  - `[[B, n_h, 1, w_local+1]]`
  - `[[B, n_h, 1, w_local+T/m_csa+1]]`
  - `[[B, n_h, 1, w_local+T/m_csa]]`
  - `[[B, n_h, 1, w_local+T/m_hca+1]]`
  - `[[B, n_h, 1, w_local+T/m_hca]]`
  - `[[B, n_h, 1, w_local]]`
  - `[[B, n_h, T+T/m_csa, d_head]]`
  - `[[B, n_h, T+T/m_hca, d_head]]`
  - `[[B, n_h, T, 1], [B, n_h, T, 1]]`
  - `[[B, n_h, T, 1]]`
  - `[[B, n_h, T, T+1]]`
  - `[[B, n_h, T, T+T/m_csa+1]]`
  - `[[B, n_h, T, T+T/m_csa]]`
  - `[[B, n_h, T, T+T/m_hca+1]]`
  - `[[B, n_h, T, T+T/m_hca]]`
  - `[[B, n_h, T, T]]`
  - `[[B, n_h, T, d_head-d_rope]]`
  - `[[B, n_h, T, d_head]]`
  - `[[B, n_h, T, d_rope/2, 2]]`
  - `[[B, n_h, T, d_rope/2]]`
  - `[[B, n_h, T, d_rope]]`
  - `[[B, n_h, d_head, T+T/m_csa]]`
  - `[[B, n_h, d_head, T+T/m_hca]]`
  - `[[B, n_h, d_head, T]]`
  - `[[B, n_h, d_head, w_local+T/m_csa]]`
  - `[[B, n_h, d_head, w_local+T/m_hca]]`
  - `[[B, n_h, d_head, w_local]]`
  - `[[B, n_h, w_local+T/m_csa, d_head]]`
  - `[[B, n_h, w_local+T/m_hca, d_head]]`
  - `[[B, n_h, w_local, d_head]]`
  - `[[]]`
  - `[[n_h, B, d_head]]`
  - `[[n_h, B, w_local+T/m_csa]]`
  - `[[n_h, B, w_local+T/m_hca]]`
  - `[[n_h, B, w_local]]`
  - `[[n_h, T+T/m_csa, d_head]]`
  - `[[n_h, T+T/m_hca, d_head]]`
  - `[[n_h, T, T+T/m_csa]]`
  - `[[n_h, T, T+T/m_hca]]`
  - `[[n_h, T, T]]`
  - `[[n_h, T, d_head]]`
  - `[[n_h, d_head, T+T/m_csa]]`
  - `[[n_h, d_head, T+T/m_hca]]`
  - `[[n_h, d_head, T]]`
  - `[[n_h, d_head, w_local+T/m_csa]]`
  - `[[n_h, d_head, w_local+T/m_hca]]`
  - `[[n_h, d_head, w_local]]`
  - `[[n_h, w_local+T/m_csa, d_head]]`
  - `[[n_h, w_local+T/m_hca, d_head]]`
  - `[[n_h, w_local, d_head]]`
- `model.layers.*.self_attn.compressor`
  - `[[B, 0, c_q]]`
  - `[[B, 0, d_head]]`
  - `[[B, 1, 1, T/m_csa+1]]`
  - `[[B, 1, 1, T/m_csa]]`
  - `[[B, 1, 1, T/m_hca]]`
  - `[[B, 1, T, 1]]`
  - `[[B, 1, T, T/m_csa+1]]`
  - `[[B, 1, T, T/m_csa]]`
  - `[[B, 1, T, T/m_hca]]`
  - `[[B, 1, T/m_csa, d_head-d_rope]]`
  - `[[B, 1, T/m_csa, d_head]]`
  - `[[B, 1, T/m_csa, d_rope/2, 2]]`
  - `[[B, 1, T/m_csa, d_rope/2]]`
  - `[[B, 1, T/m_csa, n_h]]`
  - `[[B, 1, T/m_csa]]`
  - `[[B, 1, T/m_hca, d_head-d_rope]]`
  - `[[B, 1, T/m_hca, d_head]]`
  - `[[B, 1, T/m_hca, d_rope/2, 2]]`
  - `[[B, 1, T/m_hca, d_rope/2]]`
  - `[[B, 1, T/m_hca, n_h]]`
  - `[[B, 1, T]]`
  - `[[B, 1, c_q]]`
  - `[[B, T, T/m_csa]]`
  - `[[B, T, c_q]]`
  - `[[B, T/m_csa, T/m_hca, d_head]]`
  - `[[B, T/m_csa, d_head]]`
  - `[[B, T/m_csa, d_rope/2, 1]]`
  - `[[B, T/m_csa, d_rope/2, 2]]`
  - `[[B, T/m_csa, m_csa, c_q]]`
  - `[[B, T/m_csa, m_csa, d_head]]`
  - `[[B, T/m_csa, n_h]]`
  - `[[B, T/m_csa-1, T/m_hca, d_head]]`
  - `[[B, T/m_csa-1, m_csa, c_q]]`
  - `[[B, T/m_csa-1, m_csa, d_head]]`
  - `[[B, T/m_csa]]`
  - `[[B, T/m_hca+1, d_head]]`
  - `[[B, T/m_hca, d_head]]`
  - `[[B, T/m_hca, d_rope/2, 1]]`
  - `[[B, T/m_hca, d_rope/2, 2]]`
  - `[[B, T/m_hca, m_hca, d_head]]`
  - `[[B, T/m_hca, n_h]]`
  - `[[B, T/m_hca]]`
  - `[[B, T]]`
  - `[[B, c_q, d_head]]`
  - `[[B, m_csa, c_q]]`
  - `[[B, m_csa, d_head]]`
  - `[[T/m_csa]]`
  - `[[T/m_hca]]`
- `model.layers.*.self_attn.compressor.gate_proj`
  - `[[B, 1, c_q]]`
  - `[[B, 1, d_head]]`
  - `[[B, T, c_q]]`
  - `[[B, T, d_head]]`
  - `[[B, c_q]]`
  - `[[B, d_head]]`
  - `[[B, d_model]]`
  - `[[T, c_q]]`
  - `[[T, d_head]]`
  - `[[T, d_model]]`
  - `[[d_model, c_q]]`
  - `[[d_model, d_head]]`
- `model.layers.*.self_attn.compressor.indexer`
  - `[[B, 0, 2*c_I]]`
  - `[[B, 0, c_I]]`
  - `[[B, 1, 1, n_h_I]]`
  - `[[B, 1, 1]]`
  - `[[B, 1, 2*c_I]]`
  - `[[B, 1, T, n_h_I]]`
  - `[[B, 1, T/m_csa, c_I]]`
  - `[[B, 1, T/m_csa, d_rope/2, 2]]`
  - `[[B, 1, T/m_csa, d_rope/2]]`
  - `[[B, 1, T/m_csa, n_h_I]]`
  - `[[B, 1, T/m_csa], [B, 1, T/m_csa]]`
  - `[[B, 1, T/m_csa]]`
  - `[[B, 1, d_rope/2, 1]]`
  - `[[B, 1, d_rope/2, 2]]`
  - `[[B, 1, n_h_I, c_I]]`
  - `[[B, 1, n_h_I]]`
  - `[[B, 1]]`
  - `[[B, T, 1]]`
  - `[[B, T, 2*c_I]]`
  - `[[B, T, T/m_csa], [B, T, T/m_csa]]`
  - `[[B, T, T/m_csa]]`
  - `[[B, T, d_rope/2, 1]]`
  - `[[B, T, d_rope/2, 2]]`
  - `[[B, T, n_h_I, c_I]]`
  - `[[B, T, n_h_I]]`
  - `[[B, T/m_csa, 2*m_csa, c_I]]`
  - `[[B, T/m_csa, c_I]]`
  - `[[B, T/m_csa, d_rope/2, 1]]`
  - `[[B, T/m_csa, d_rope/2, 2]]`
  - `[[B, T/m_csa, m_csa, 2*c_I]]`
  - `[[B, T/m_csa, m_csa, c_I]]`
  - `[[B, T/m_csa, n_h_I]]`
  - `[[B, T/m_csa-1, 2*m_csa, c_I]]`
  - `[[B, T/m_csa-1, m_csa, 2*c_I]]`
  - `[[B, T/m_csa-1, m_csa, c_I]]`
  - `[[B, T/m_csa]]`
  - `[[B, T]]`
  - `[[B, m_csa, 2*c_I]]`
  - `[[B, m_csa, c_I]]`
  - `[[B, n_h_I, 1, c_I]]`
  - `[[B, n_h_I, 1, d_rope/2, 2]]`
  - `[[B, n_h_I, 1, d_rope/2]]`
  - `[[B, n_h_I, 1, n_h]]`
  - `[[B, n_h_I, T, c_I]]`
  - `[[B, n_h_I, T, d_rope/2, 2]]`
  - `[[B, n_h_I, T, d_rope/2]]`
  - `[[B, n_h_I, T, n_h]]`
  - `[[T/m_csa]]`
- `model.layers.*.self_attn.compressor.indexer.gate_proj`
  - `[[B, 1, 2*c_I]]`
  - `[[B, 2*c_I]]`
  - `[[B, T, 2*c_I]]`
  - `[[B, d_model]]`
  - `[[T, 2*c_I]]`
  - `[[T, d_model]]`
  - `[[d_model, 2*c_I]]`
- `model.layers.*.self_attn.compressor.indexer.kv_norm`
  - `[[B, T/m_csa, 1]]`
  - `[[B, T/m_csa, c_I]]`
- `model.layers.*.self_attn.compressor.indexer.kv_proj`
  - `[[B, 1, 2*c_I]]`
  - `[[B, 2*c_I]]`
  - `[[B, T, 2*c_I]]`
  - `[[B, d_model]]`
  - `[[T, 2*c_I]]`
  - `[[T, d_model]]`
  - `[[d_model, 2*c_I]]`
- `model.layers.*.self_attn.compressor.indexer.q_b_proj`
  - `[[B, 1, n_h_I*c_I]]`
  - `[[B, T, n_h_I*c_I]]`
  - `[[B, c_q]]`
  - `[[B, n_h_I*c_I]]`
  - `[[T, c_q]]`
  - `[[T, n_h_I*c_I]]`
  - `[[c_q, n_h_I*c_I]]`
- `model.layers.*.self_attn.compressor.indexer.rotary_emb`
  - `[[B, 1, 1]]`
  - `[[B, 1, T/m_csa]]`
  - `[[B, 1, T]]`
  - `[[B, 1, d_rope/2]]`
  - `[[B, T, d_rope/2]]`
  - `[[B, T/m_csa, d_rope/2]]`
  - `[[B, d_rope/2, 1]]`
  - `[[B, d_rope/2, T/m_csa]]`
  - `[[B, d_rope/2, T]]`
  - `[[B, d_rope/2]]`
- `model.layers.*.self_attn.compressor.indexer.scorer`
  - `[[B, 1, T/m_csa]]`
  - `[[B, 1, c_I, T/m_csa]]`
  - `[[B, 1, n_h_I, 1]]`
  - `[[B, 1, n_h_I, T/m_csa]]`
  - `[[B, 1, n_h_I, c_I]]`
  - `[[B, 1, n_h_I]]`
  - `[[B, T, T/m_csa]]`
  - `[[B, T, c_I, T/m_csa]]`
  - `[[B, T, n_h_I, 1]]`
  - `[[B, T, n_h_I, T/m_csa]]`
  - `[[B, T, n_h_I, c_I]]`
  - `[[B, T, n_h_I]]`
  - `[[B, c_I, T/m_csa]]`
  - `[[B, n_h_I, T/m_csa]]`
  - `[[B, n_h_I, c_I]]`
  - `[[T, c_I, T/m_csa]]`
  - `[[T, n_h_I, T/m_csa]]`
  - `[[T, n_h_I, c_I]]`
- `model.layers.*.self_attn.compressor.indexer.scorer.weights_proj`
  - `[[B, 1, n_h_I]]`
  - `[[B, T, n_h_I]]`
  - `[[B, d_model]]`
  - `[[B, n_h_I]]`
  - `[[T, d_model]]`
  - `[[T, n_h_I]]`
  - `[[d_model, n_h_I]]`
- `model.layers.*.self_attn.compressor.kv_norm`
  - `[[B, T/m_csa, 1]]`
  - `[[B, T/m_csa, d_head]]`
  - `[[B, T/m_hca, 1]]`
  - `[[B, T/m_hca, d_head]]`
- `model.layers.*.self_attn.compressor.kv_proj`
  - `[[B, 1, c_q]]`
  - `[[B, 1, d_head]]`
  - `[[B, T, c_q]]`
  - `[[B, T, d_head]]`
  - `[[B, c_q]]`
  - `[[B, d_head]]`
  - `[[B, d_model]]`
  - `[[T, c_q]]`
  - `[[T, d_head]]`
  - `[[T, d_model]]`
  - `[[d_model, c_q]]`
  - `[[d_model, d_head]]`
- `model.layers.*.self_attn.compressor.rotary_emb`
  - `[[B, 1, T/m_csa]]`
  - `[[B, 1, T/m_hca]]`
  - `[[B, T/m_csa, d_rope/2]]`
  - `[[B, T/m_hca, d_rope/2]]`
  - `[[B, d_rope/2, 1]]`
  - `[[B, d_rope/2, T/m_csa]]`
  - `[[B, d_rope/2, T/m_hca]]`
  - `[[B, d_rope/2]]`
- `model.layers.*.self_attn.kv_norm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_head]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_head]]`
- `model.layers.*.self_attn.kv_proj`
  - `[[B, 1, d_head]]`
  - `[[B, T, d_head]]`
  - `[[B, d_head]]`
  - `[[B, d_model]]`
  - `[[T, d_head]]`
  - `[[T, d_model]]`
  - `[[d_model, d_head]]`
- `model.layers.*.self_attn.o_a_proj`
  - `[[B, 1, g_o, d_g]]`
  - `[[B, T, g_o, d_g]]`
  - `[[B, T/m_hca, d_model]]`
  - `[[B, g_o, d_g]]`
  - `[[T, T/m_hca, d_model]]`
  - `[[T, g_o, d_g]]`
  - `[[g_o, B, d_g]]`
  - `[[g_o, B, d_model]]`
  - `[[g_o, T, d_g]]`
  - `[[g_o, T, d_model]]`
  - `[[g_o, d_g, d_model]]`
  - `[[g_o, d_model, d_g]]`
- `model.layers.*.self_attn.o_b_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[B, g_o*d_g]]`
  - `[[T, d_model]]`
  - `[[T, g_o*d_g]]`
  - `[[g_o*d_g, d_model]]`
- `model.layers.*.self_attn.q_a_norm`
  - `[[B, 1, 1]]`
  - `[[B, 1, c_q]]`
  - `[[B, T, 1]]`
  - `[[B, T, c_q]]`
- `model.layers.*.self_attn.q_a_proj`
  - `[[B, 1, c_q]]`
  - `[[B, T, c_q]]`
  - `[[B, c_q]]`
  - `[[B, d_model]]`
  - `[[T, c_q]]`
  - `[[T, d_model]]`
  - `[[d_model, c_q]]`
- `model.layers.*.self_attn.q_b_norm`
  - `[[B, n_h, 1, 1]]`
  - `[[B, n_h, 1, d_head]]`
  - `[[B, n_h, T, 1]]`
  - `[[B, n_h, T, d_head]]`
- `model.layers.*.self_attn.q_b_proj`
  - `[[B, 1, n_h*d_head]]`
  - `[[B, T, n_h*d_head]]`
  - `[[B, c_q]]`
  - `[[B, n_h*d_head]]`
  - `[[T, c_q]]`
  - `[[T, n_h*d_head]]`
  - `[[c_q, n_h*d_head]]`
- `model.layers.0`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.1`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.10`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.11`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.12`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.13`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.14`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.15`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.16`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.17`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.18`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.19`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.2`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.20`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.21`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.22`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.23`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.24`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.25`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.26`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.27`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.28`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.29`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.3`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.30`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.31`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.32`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.33`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.34`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.35`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.36`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.37`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.38`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.39`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.4`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.40`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.41`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.42`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.5`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.6`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.7`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.8`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.layers.9`
  - `[[B, 1, 1, d_model]]`
  - `[[B, 1, n_hc, 1]]`
  - `[[B, 1, n_hc, d_model]]`
  - `[[B, 1, n_hc, n_hc]]`
  - `[[B, 1, n_hc]]`
  - `[[B, T, 1, d_model]]`
  - `[[B, T, n_hc, 1]]`
  - `[[B, T, n_hc, d_model]]`
  - `[[B, T, n_hc, n_hc]]`
  - `[[B, T, n_hc]]`
  - `[[B, n_hc, d_model]]`
  - `[[B, n_hc, n_hc]]`
  - `[[T, n_hc, d_model]]`
  - `[[T, n_hc, n_hc]]`
- `model.norm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
- `model.rotary_emb`
  - `[[B, 1, 1]]`
  - `[[B, 1, T]]`
  - `[[B, 1, d_rope/2]]`
  - `[[B, T, d_rope/2]]`
  - `[[B, d_rope/2, 1]]`
  - `[[B, d_rope/2, T]]`
  - `[[B, d_rope/2]]`

## 이 의뢰서를 처리하는 법

`review/prompt.md` 를 LLM 에 넘기고 이 모델을 지정한다. 판정 4종과 근거 요건, 결과를 어디에 어떤 형식으로 쓰는지는 전부 `review/` 안에 있다.
