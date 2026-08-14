# 검토 의뢰서 — LiquidAI/LFM2-8B-A1B

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `lfm2_moe`
- 판단 필요: **1건**

## 증거 — 이미 받아둔 실제 소스

- `develop/sources/modeling_lfm2_moe.py` — 있음, 이 파일을 열어서 판정한다
- `develop/sources/configuration_lfm2_moe.py` — 있음, 이 파일을 열어서 판정한다

- 온라인 원본: https://github.com/huggingface/transformers/tree/main/src/transformers/models/lfm2_moe

그 밖의 재료: `full/review.md`(리뷰 패킷 — shape 별 실제 행 표본), `structure.yaml`(이 모델의 심볼 표), `full/<phase>.csv`(전체 operator 표).

## 판단이 필요한 것

### 4. 규칙 없이 산술로 지은 이름

값이 맞아떨어져서 붙인 이름이다. 산술적으로 참이어도 틀린 이름일 수 있으므로 (예: RoPE 절반 차원) 소스에서 확인이 필요하다.

- `3*d_model` in `model.layers.*.conv.in_proj (레이어 12개)` — heur_multiple, 290축

### 0. 규칙이 끝내지 못한 축 — **여기부터 답한다**

값으로는 결정할 수 없어 파이프라인이 판단을 넘긴 자리다. 세 가지뿐이다:
`tie`(두 심볼이 같은 값이라 관례로 골랐다) · `heur`(등록 규칙이 없어 산술로 지어냈다) · `bare`(이름을 못 붙였는데 크기가 커서 진짜 차원일 수 있다).

**답이 나오면 `override_stub` 을 채워 `rules/label_overrides.yaml` 에 넣는다.** `spread: class` 라 그 축이 지나는 모든 자리가 한 번에 바뀐다 — 모듈 경계에서 멈추지 않는다(그것이 예전에 교정을 막던 유일한 이유였다).

**값이 같은 심볼이 여럿이면 값으로는 영원히 못 가른다. shape 안의 위치가 말해 준다** — `[B, n_h, T, d_head]` 처럼. 표본 shape 을 같이 싣는 이유다.

| 왜 | 모듈 | 크기 | 지금 이름 | 후보 | 축 위치 | 표본 shape | 축 수 |
|---|---|---|---|---|---|---|---|
| `heur` | `model.layers.*.conv` | 6144 | `3*d_model` | — | 1/2 | `[d_model, 3*d_model]  (축 1)` | 108 |
| `heur` | `model.layers.*.conv` | 6144 | `3*d_model` | — | 1/3 | `[B, 3*d_model, T]  (축 1)` | 36 |
| `heur` | `model.layers.*.conv.in_proj` | 6144 | `3*d_model` | — | 0/2 | `[3*d_model, d_model]  (축 0)` | 18 |

초안(그대로 복사해 `to` 와 `source` 만 채운다):

```yaml
  - model: LiquidAI__LFM2-8B-A1B
    module: 'conv$'
    spread: class
    from: 3*d_model
    to: <소스가 말하는 이름>
    expect: 6144
    source: <modeling_*.py:줄 인용>
  - model: LiquidAI__LFM2-8B-A1B
    module: 'conv$'
    spread: class
    from: 3*d_model
    to: <소스가 말하는 이름>
    expect: 6144
    source: <modeling_*.py:줄 인용>
  - model: LiquidAI__LFM2-8B-A1B
    module: 'conv\.in_proj$'
    spread: class
    from: 3*d_model
    to: <소스가 말하는 이름>
    expect: 6144
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

고유 행 64개.

| phase | 모듈 | op | input_shape | weight_shape | output_shape |
|---|---|---|---|---|---|
| prefill | `model.embed_tokens` | embedding | `[['V', 'd_model'], ['B', 'T']]` | `['V', 'd_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.pos_emb` | batched_matmul | `[['B', 'd_head/2', '1'], ['B', '1', 'T']]` | `None` | `[['B', 'd_head/2', 'T']]` |
| prefill | `model.layers.*.operator_norm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.conv.in_proj` | matmul | `[['T', 'd_model'], ['d_model', '3*d_model']]` | `['3*d_model', 'd_model']` | `[['T', '3*d_model']]` |
| prefill | `model.layers.*.conv.conv` | conv1d | `[['B', 'd_model', 'T'], ['d_model', '1', 'd_conv']]` | `['d_model', '1', 'd_conv']` | `[['B', 'd_model', 'T+d_conv-1']]` |
| prefill | `model.layers.*.conv.out_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_model']]` | `['d_model', 'd_model']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_add | `[['B', 'T', 'd_model'], ['B', 'T', 'd_model']]` | `None` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.ffn_norm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.feed_forward.w1` | matmul | `[['T', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['T', 'd_ff']]` |
| prefill | `model.layers.*.feed_forward` | silu | `[['B', 'T', 'd_ff']]` | `None` | `[['B', 'T', 'd_ff']]` |
| prefill | `model.layers.*.feed_forward.w3` | matmul | `[['T', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['T', 'd_ff']]` |
| prefill | `model.layers.*.feed_forward` | elementwise_mul | `[['B', 'T', 'd_ff'], ['B', 'T', 'd_ff']]` | `None` | `[['B', 'T', 'd_ff']]` |
| prefill | `model.layers.*.feed_forward.w2` | matmul | `[['T', 'd_ff'], ['d_ff', 'd_model']]` | `['d_model', 'd_ff']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*.self_attn.q_proj` | matmul | `[['T', 'd_model'], ['d_model', 'n_h*d_head']]` | `['n_h*d_head', 'd_model']` | `[['T', 'n_h*d_head']]` |
| prefill | `model.layers.*.self_attn.q_layernorm` | rmsnorm | `[['B', 'T', 'n_h', 'd_head']]` | `['d_head']` | `[['B', 'T', 'n_h', 'd_head']]` |
| prefill | `model.layers.*.self_attn.k_proj` | matmul | `[['T', 'd_model'], ['d_model', 'n_kv*d_head']]` | `['n_kv*d_head', 'd_model']` | `[['T', 'n_kv*d_head']]` |
| prefill | `model.layers.*.self_attn.k_layernorm` | rmsnorm | `[['B', 'T', 'n_kv', 'd_head']]` | `['d_head']` | `[['B', 'T', 'n_kv', 'd_head']]` |
| prefill | `model.layers.*.self_attn.v_proj` | matmul | `[['T', 'd_model'], ['d_model', 'n_kv*d_head']]` | `['n_kv*d_head', 'd_model']` | `[['T', 'n_kv*d_head']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'T', 'd_head'], ['n_h', 'd_head', 'T']]` | `None` | `[['n_h', 'T', 'T']]` |
| prefill | `model.layers.*.self_attn` | softmax | `[['B', 'n_h', 'T', 'T']]` | `None` | `[['B', 'n_h', 'T', 'T']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'T', 'T'], ['n_h', 'T', 'd_head']]` | `None` | `[['n_h', 'T', 'd_head']]` |
| prefill | `model.layers.*.self_attn.out_proj` | matmul | `[['T', 'n_h*d_head'], ['n_h*d_head', 'd_model']]` | `['d_model', 'n_h*d_head']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*.feed_forward.gate` | matmul | `[['T', 'd_model'], ['d_model', 'E']]` | `['E', 'd_model']` | `[['T', 'E']]` |
| prefill | `model.layers.*.feed_forward.gate` | sigmoid | `[['T', 'E']]` | `None` | `[['T', 'E']]` |
| prefill | `model.layers.*.feed_forward.experts` | grouped_matmul | `[['k*T', 'd_model'], ['E', 'd_model', '2*d_moe'], ['E']]` | `['E', '2*d_moe', 'd_model']` | `[['k*T', '2*d_moe']]` |
| prefill | `model.layers.*.feed_forward.experts` | silu | `[['k*T', 'd_moe']]` | `None` | `[['k*T', 'd_moe']]` |
| prefill | `model.layers.*.feed_forward.experts` | elementwise_mul | `[['k*T', 'd_moe'], ['k*T', 'd_moe']]` | `None` | `[['k*T', 'd_moe']]` |
| prefill | `model.layers.*.feed_forward.experts` | grouped_matmul | `[['k*T', 'd_moe'], ['E', 'd_moe', 'd_model'], ['E']]` | `['E', 'd_model', 'd_moe']` | `[['k*T', 'd_model']]` |
| prefill | `model.layers.*.feed_forward.experts` | elementwise_mul | `[['k*T', 'd_model'], ['k*T', 'B']]` | `None` | `[['k*T', 'd_model']]` |
| prefill | `model.layers.*.feed_forward.experts` | sum | `[['T', 'k', 'd_model']]` | `None` | `[['T', 'd_model']]` |
| prefill | `model.embedding_norm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `lm_head` | matmul | `[['T', 'd_model'], ['d_model', 'V']]` | `['V', 'd_model']` | `[['T', 'V']]` |
| decode | `model.embed_tokens` | embedding | `[['V', 'd_model'], ['B', '1']]` | `['V', 'd_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.pos_emb` | batched_matmul | `[['B', 'd_head/2', '1'], ['B', '1', '1']]` | `None` | `[['B', 'd_head/2', '1']]` |
| decode | `model.layers.*.operator_norm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.conv.in_proj` | matmul | `[['B', 'd_model'], ['d_model', '3*d_model']]` | `['3*d_model', 'd_model']` | `[['B', '3*d_model']]` |
| decode | `model.layers.*.conv` | sum | `[['B', 'd_model', 'd_conv']]` | `None` | `[['B', 'd_model']]` |
| decode | `model.layers.*.conv.out_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_model']]` | `['d_model', 'd_model']` | `[['B', 'd_model']]` |
| decode | `model.layers.*` | elementwise_add | `[['B', '1', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.ffn_norm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.feed_forward.w1` | matmul | `[['B', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['B', 'd_ff']]` |
| decode | `model.layers.*.feed_forward` | silu | `[['B', '1', 'd_ff']]` | `None` | `[['B', '1', 'd_ff']]` |
| decode | `model.layers.*.feed_forward.w3` | matmul | `[['B', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['B', 'd_ff']]` |
| decode | `model.layers.*.feed_forward` | elementwise_mul | `[['B', '1', 'd_ff'], ['B', '1', 'd_ff']]` | `None` | `[['B', '1', 'd_ff']]` |
| decode | `model.layers.*.feed_forward.w2` | matmul | `[['B', 'd_ff'], ['d_ff', 'd_model']]` | `['d_model', 'd_ff']` | `[['B', 'd_model']]` |
| decode | `model.layers.*.self_attn.q_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_h*d_head']]` | `['n_h*d_head', 'd_model']` | `[['B', 'n_h*d_head']]` |
| decode | `model.layers.*.self_attn.q_layernorm` | rmsnorm | `[['B', '1', 'n_h', 'd_head']]` | `['d_head']` | `[['B', '1', 'n_h', 'd_head']]` |
| decode | `model.layers.*.self_attn.k_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_kv*d_head']]` | `['n_kv*d_head', 'd_model']` | `[['B', 'n_kv*d_head']]` |
| decode | `model.layers.*.self_attn.k_layernorm` | rmsnorm | `[['B', '1', 'n_kv', 'd_head']]` | `['d_head']` | `[['B', '1', 'n_kv', 'd_head']]` |
| decode | `model.layers.*.self_attn.v_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_kv*d_head']]` | `['n_kv*d_head', 'd_model']` | `[['B', 'n_kv*d_head']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'B', 'd_head'], ['n_h', 'd_head', 'T+1']]` | `None` | `[['n_h', 'B', 'T+1']]` |
| decode | `model.layers.*.self_attn` | softmax | `[['B', 'n_h', '1', 'T+1']]` | `None` | `[['B', 'n_h', '1', 'T+1']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'B', 'T+1'], ['n_h', 'T+1', 'd_head']]` | `None` | `[['n_h', 'B', 'd_head']]` |
| decode | `model.layers.*.self_attn.out_proj` | matmul | `[['B', 'n_h*d_head'], ['n_h*d_head', 'd_model']]` | `['d_model', 'n_h*d_head']` | `[['B', 'd_model']]` |
| decode | `model.layers.*.feed_forward.gate` | matmul | `[['B', 'd_model'], ['d_model', 'E']]` | `['E', 'd_model']` | `[['B', 'E']]` |
| decode | `model.layers.*.feed_forward.gate` | sigmoid | `[['B', 'E']]` | `None` | `[['B', 'E']]` |
| decode | `model.layers.*.feed_forward.experts` | grouped_matmul | `[['k', 'd_model'], ['E', 'd_model', '2*d_moe'], ['E']]` | `['E', '2*d_moe', 'd_model']` | `[['k', '2*d_moe']]` |
| decode | `model.layers.*.feed_forward.experts` | silu | `[['k', 'd_moe']]` | `None` | `[['k', 'd_moe']]` |
| decode | `model.layers.*.feed_forward.experts` | elementwise_mul | `[['k', 'd_moe'], ['k', 'd_moe']]` | `None` | `[['k', 'd_moe']]` |
| decode | `model.layers.*.feed_forward.experts` | grouped_matmul | `[['k', 'd_moe'], ['E', 'd_moe', 'd_model'], ['E']]` | `['E', 'd_model', 'd_moe']` | `[['k', 'd_model']]` |
| decode | `model.layers.*.feed_forward.experts` | elementwise_mul | `[['k', 'd_model'], ['k', 'B']]` | `None` | `[['k', 'd_model']]` |
| decode | `model.layers.*.feed_forward.experts` | sum | `[['B', 'k', 'd_model']]` | `None` | `[['B', 'd_model']]` |
| decode | `model.embedding_norm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `lm_head` | matmul | `[['B', 'd_model'], ['d_model', 'V']]` | `['V', 'd_model']` | `[['B', 'V']]` |

## 전수 점검 — 이 모델이 쓰는 이름 전부

위 절이 '풀리지 않은 것'이라면 여기는 **전부**다. 규칙이 자신 있게 붙인 이름도 틀릴 수 있고, 그런 건 미결 목록에 절대 오르지 않는다. 한 줄씩 읽고 **그 모듈에서 그 이름이 말이 되는지** 보라.

### A. 붙은 이름 전부 (22종)

| 라벨 | 값 | 나타나는 모듈 | 축 수 |
|---|---|---|---|
| `B` |  | `model.layers.*.self_attn`, `model.layers.*.conv`, `model.layers.*.operator_norm`, `model.layers.*.ffn_norm` 외 45개 | 6368 |
| `d_model` | 2048 | `model.layers.*.feed_forward.experts`, `model.layers.*.conv`, `model.layers.*.operator_norm`, `model.layers.*.conv.out_proj` 외 40개 | 5462 |
| `T` |  | `model.layers.*.self_attn`, `model.layers.*.feed_forward.gate`, `model.layers.*.operator_norm`, `model.layers.*.ffn_norm` 외 45개 | 3472 |
| `k` | 4 | `model.layers.*.feed_forward.experts`, `model.layers.*.feed_forward.gate` | 1782 |
| `d_head` | 64 | `model.layers.*.self_attn`, `model.layers.*.self_attn.q_layernorm`, `model.layers.*.self_attn.k_layernorm`, `model.pos_emb` | 1298 |
| `E` | 32 | `model.layers.*.feed_forward.experts`, `model.layers.*.feed_forward.gate` | 1232 |
| `k*T` |  | `model.layers.*.feed_forward.experts` | 1210 |
| `n_h` | 32 | `model.layers.*.self_attn`, `model.layers.*.self_attn.q_layernorm` | 960 |
| `n_kv` | 8 | `model.layers.*.self_attn`, `model.layers.*.self_attn.k_layernorm` | 696 |
| `d_moe` | 1792 | `model.layers.*.feed_forward.experts` | 572 |
| `d_conv` | 3 | `model.layers.*.conv`, `model.layers.*.conv.conv` | 432 |
| `3*d_model` |  | `model.layers.*.conv.in_proj`, `model.layers.*.conv` | 396 |
| `T+1` |  | `model.layers.*.self_attn`, `model` | 309 |
| `2*d_moe` |  | `model.layers.*.feed_forward.experts` | 308 |
| `n_h*d_head` |  | `model.layers.*.self_attn.q_proj`, `model.layers.*.self_attn.out_proj`, `model.layers.*.self_attn` | 216 |
| `n_kv*d_head` |  | `model.layers.*.self_attn.k_proj`, `model.layers.*.self_attn.v_proj`, `model.layers.*.self_attn` | 216 |
| `d_head/2` |  | `model.layers.*.self_attn`, `model.pos_emb` | 180 |
| `d_ff` | 7168 | `model.layers.*.feed_forward.w1`, `model.layers.*.feed_forward.w3`, `model.layers.*.feed_forward.w2`, `model.layers.*.feed_forward` | 116 |
| `n_h/n_kv` |  | `model.layers.*.self_attn` | 96 |
| `d_conv+1` |  | `model.layers.*.conv` | 54 |
| `T+d_conv-1` |  | `model.layers.*.conv.conv`, `model.layers.*.conv` | 36 |
| `V` | 65536 | `lm_head`, `model.embed_tokens` | 20 |

### B. 이름 없이 남은 정수 전부 (0쌍)

**여기가 필터가 못 보던 자리다.** 정수가 남는 것 자체는 정상이다(루프 인덱스, 피연산자 개수, 브로드캐스트 축). 문제는 **이름이 있어야 하는데 없는 경우**이고, 마지막 열이 그 신호다 — 이 모델의 심볼과 값이 같다면 스코프가 그 모듈을 못 덮고 있을 수 있다. 실제로 `n_hc`(=4)가 그렇게 정수로 남아 있었다.

| 모듈 | 정수 | 축 수 | 같은 값의 심볼 |
|---|---|---|---|

### C. 모듈이 내는 출력 shape 전부 (49개 모듈 / 246종)

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
- `model.embed_tokens`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.embedding_norm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
- `model.layers.*.conv`
  - `[[B, 1, d_model]]`
  - `[[B, 3*d_model, 1]]`
  - `[[B, 3*d_model, T]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model, 1], [B, d_model, 1], [B, d_model, 1]]`
  - `[[B, d_model, 1]]`
  - `[[B, d_model, T], [B, d_model, T], [B, d_model, T]]`
  - `[[B, d_model, T]]`
  - `[[B, d_model, d_conv+1]]`
  - `[[B, d_model, d_conv]]`
  - `[[B, d_model]]`
  - `[[d_model, d_conv]]`
- `model.layers.*.conv.conv`
  - `[[B, d_model, T+d_conv-1]]`
- `model.layers.*.conv.in_proj`
  - `[[B, 1, 3*d_model]]`
  - `[[B, 3*d_model]]`
  - `[[B, T, 3*d_model]]`
  - `[[B, d_model]]`
  - `[[T, 3*d_model]]`
  - `[[T, d_model]]`
  - `[[d_model, 3*d_model]]`
- `model.layers.*.conv.out_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[T, d_model]]`
  - `[[d_model, d_model]]`
- `model.layers.*.feed_forward`
  - `[[B, 1, d_ff]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_ff]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[T, d_model]]`
- `model.layers.*.feed_forward.experts`
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
- `model.layers.*.feed_forward.gate`
  - `[[B, 1]]`
  - `[[B, E]]`
  - `[[B, k], [B, k]]`
  - `[[B, k]]`
  - `[[T, 1]]`
  - `[[T, E]]`
  - `[[T, k], [T, k]]`
  - `[[T, k]]`
  - `[[d_model, E]]`
- `model.layers.*.feed_forward.w1`
  - `[[B, 1, d_ff]]`
  - `[[B, T, d_ff]]`
  - `[[B, d_ff]]`
  - `[[B, d_model]]`
  - `[[T, d_ff]]`
  - `[[T, d_model]]`
  - `[[d_model, d_ff]]`
- `model.layers.*.feed_forward.w2`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_ff]]`
  - `[[B, d_model]]`
  - `[[T, d_ff]]`
  - `[[T, d_model]]`
  - `[[d_ff, d_model]]`
- `model.layers.*.feed_forward.w3`
  - `[[B, 1, d_ff]]`
  - `[[B, T, d_ff]]`
  - `[[B, d_ff]]`
  - `[[B, d_model]]`
  - `[[T, d_ff]]`
  - `[[T, d_model]]`
  - `[[d_model, d_ff]]`
- `model.layers.*.ffn_norm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
- `model.layers.*.operator_norm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
- `model.layers.*.self_attn`
  - `[[B, 1, 1, d_head]]`
  - `[[B, 1, T, d_head]]`
  - `[[B, 1, n_h*d_head]]`
  - `[[B, 1, n_h, d_head]]`
  - `[[B, 1, n_kv, d_head]]`
  - `[[B, T, n_h*d_head]]`
  - `[[B, T, n_h, d_head]]`
  - `[[B, T, n_kv, d_head]]`
  - `[[B, n_h, 1, T+1]]`
  - `[[B, n_h, 1, d_head/2]]`
  - `[[B, n_h, 1, d_head]]`
  - `[[B, n_h, T+1, d_head]]`
  - `[[B, n_h, T, T]]`
  - `[[B, n_h, T, d_head/2]]`
  - `[[B, n_h, T, d_head]]`
  - `[[B, n_h, d_head, T+1]]`
  - `[[B, n_h, d_head, T]]`
  - `[[B, n_kv, 1, T+1, d_head]]`
  - `[[B, n_kv, 1, T, d_head]]`
  - `[[B, n_kv, 1, d_head/2]]`
  - `[[B, n_kv, 1, d_head]]`
  - `[[B, n_kv, T+1, d_head]]`
  - `[[B, n_kv, T, d_head/2]]`
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
- `model.layers.*.self_attn.k_layernorm`
  - `[[B, 1, n_kv, 1]]`
  - `[[B, 1, n_kv, d_head]]`
  - `[[B, T, n_kv, 1]]`
  - `[[B, T, n_kv, d_head]]`
- `model.layers.*.self_attn.k_proj`
  - `[[B, 1, n_kv*d_head]]`
  - `[[B, T, n_kv*d_head]]`
  - `[[B, d_model]]`
  - `[[B, n_kv*d_head]]`
  - `[[T, d_model]]`
  - `[[T, n_kv*d_head]]`
  - `[[d_model, n_kv*d_head]]`
- `model.layers.*.self_attn.out_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[B, n_h*d_head]]`
  - `[[T, d_model]]`
  - `[[T, n_h*d_head]]`
  - `[[n_h*d_head, d_model]]`
- `model.layers.*.self_attn.q_layernorm`
  - `[[B, 1, n_h, 1]]`
  - `[[B, 1, n_h, d_head]]`
  - `[[B, T, n_h, 1]]`
  - `[[B, T, n_h, d_head]]`
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
- `model.pos_emb`
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
