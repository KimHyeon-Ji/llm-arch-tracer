# 배치 A 판정 — Codex CSV/JSONL 라벨 주장 검증

대상: MiniMaxAI__MiniMax-M2, NX-AI__xLSTM-7b, Qwen__Qwen3-Next-80B-A3B-Instruct,
allenai__OLMo-2-1124-7B-Instruct
근거 소스: **설치된 transformers 5.14.1** (`.venv/Lib/site-packages/transformers/...`)
판정 시작 2026-09-03. 진행하며 append.

---

## A-1 · MiniMax-M2 · RoPE 절반 축 `n_h+2*n_kv` → `d_head/2` — **CONFIRMED**

- **CSV 자리**: `models/MiniMaxAI__MiniMax-M2/full/prefill.csv` op_id 91~94,
  `module_path=model.layers.N.self_attn`, op `slice`/`neg`/`concat`.
- **현재**: op 91 `input [[B, n_h, T, d_head]]` → `output [[B, n_h, T, n_h+2*n_kv]]`
- **소스**: `modeling_minimax_m2.py:288-292`
  ```python
  def rotate_half(x):
      x1 = x[..., : x.shape[-1] // 2]
      x2 = x[..., x.shape[-1] // 2 :]
      return torch.cat((-x2, x1), dim=-1)
  ```
  잘리는 축은 per-head 텐서의 마지막 축의 절반 = `d_head/2`.
- **값 충돌**: `d_head/2` = 128/2 = 64, `n_h+2*n_kv` = 48+2·8 = 64. 숫자가 같아 수치
  검증으로는 안 잡힘.
- **자체 모순**: 입력 축이 `d_head` 인데 그것을 slice 한 결과가 head 개수 식이 될 수 없음.
- **분류**: `heuristic-fabricated-labels` (RoPE 절반 축을 산술적으로만 맞는 이름으로 지어냄).

## A-2 · MiniMax-M2 · experts gate-up weight 축 `d_model` → `2*d_moe` — **CONFIRMED**

- **CSV 자리**: 같은 파일 op_id 174(`transpose`), 175(`grouped_matmul`),
  `module_path=model.layers.N.mlp.experts`, weight = `experts.gate_up_proj`.
- **현재**: `weight_shape [E, d_model, d_model]`
- **소스**: `modeling_minimax_m2.py:76`
  ```python
  self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
  ```
  → 올바른 표기는 `[E, 2*d_moe, d_model]`.
- **값 충돌**: `2*d_moe` = 2·1536 = 3072 = `d_model`.
- **자체 모순 (결정적)**: 같은 행 175의 `output_shape` 는 **이미** `[k*T, 2*d_moe]` 로
  정확히 렌더링돼 있음. 즉 같은 op 안에서 weight 축과 output 축이 서로 다른 이름을
  쓰고 있어, 산출물 내부에서 이미 불일치.
- **부수 효과**: op 174 transpose 가 `[E,d_model,d_model]`→`[E,d_model,d_model]` 로
  아무것도 안 바꾼 것처럼 보임. 교정 시 `[E,2*d_moe,d_model]`→`[E,d_model,2*d_moe]`.

## A-8 · OLMo-2-7B · `k_proj`/`v_proj` 폭 `n_h*d_head` → `n_kv*d_head` — **CONFIRMED**

- **CSV 자리**: `models/allenai__OLMo-2-1124-7B-Instruct/full/prefill.csv`
  op_id 1110~1113(k_proj), 1122~1125(v_proj) 외 전 레이어.
  현재 `weight_shape [n_h*d_head, d_model]`, `output [[B, T, n_h*d_head]]`.
- **소스**: `modeling_olmo2.py:220-227`
  ```python
  self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, ...)
  self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, ...)
  self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, ...)
  ```
  k/v 는 `num_key_value_heads` 를 읽는다 → `n_kv*d_head`. (`module-field-membership-check`)
- **값 충돌**: 이 모델은 MHA(`n_h = n_kv = 32`)라 둘 다 4096. 숫자로는 영원히 안 잡힘.
- **Codex 범위보다 넓음 — 내가 확장 발견**: `k_norm` 도 같은 계열이다.
  `modeling_olmo2.py:233` `self.k_norm = Olmo2RMSNorm(config.num_key_value_heads * self.head_dim, ...)`
  인데 CSV op_id 34~37 등 `self_attn.k_norm` 행이 전부 `n_h*d_head` 로 렌더링됨.
  (`q_norm` 은 232행에서 `num_attention_heads` 이므로 `n_h*d_head` 가 맞다 — 건드리지 말 것.)
- **영향 행 수**: k_proj/v_proj/k_norm 에 `n_h*d_head` 가 붙은 행 = prefill 384 + decode 384
  = **768행**.

## A-3 · Qwen3-Next · `linear_attn.in_proj_ba` 폭 `d_chunk` → `2*n_h_lin_v` — **CONFIRMED**

- **CSV 자리**: `models/Qwen__Qwen3-Next-80B-A3B-Instruct/full/prefill.csv` op_id 34~37.
  `weight_shape [d_chunk, d_model]`, `output [[B, T, d_chunk]]`.
- **소스**: `modeling_qwen3_next.py:528,531`
  ```python
  projection_size_ba = self.num_v_heads * 2
  self.in_proj_ba = nn.Linear(self.hidden_size, projection_size_ba, bias=False)
  ```
  → `2*n_h_lin_v` = 2·32 = 64.
- **값 충돌**: `d_chunk` = 64 로 동일. `in_proj_ba` 는 chunk 크기와 아무 관계가 없다
  (β/α 게이트를 v-head 당 2개 만드는 투영).

## A-4 · Qwen3-Next · fused-Q `split` 마지막 축 `n_kv*d_head` → `2*d_head` — **CONFIRMED**

- **CSV 자리**: 같은 파일 op_id 3386 `self_attn` `split`:
  `input [[B, T, n_h, n_kv*d_head]]` → `output [[B,T,n_h,d_head], [B,T,n_h,d_head]]`
- **소스**: `modeling_qwen3_next.py:268-270, 296-298`
  ```python
  self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim * 2, ...)
  query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
  ```
  view 의 마지막 축은 `head_dim*2` = `2*d_head`.
- **값 충돌**: `2*d_head` = 2·256 = 512, `n_kv*d_head` = 2·256 = 512.
- **자체 모순**: 같은 행에서 `d_head` 두 조각으로 쪼개지는 축이 kv-head 식일 수 없음.
  더구나 op 3381~3384 의 `q_proj` 는 **이미** `2*n_h*d_head` 로 정확히 렌더링돼 있다.

## A-5 · Qwen3-Next · `mlp.experts` `E` ↔ `d_moe` — **CONFIRMED (Codex 범위보다 넓음)**

- **소스**: `modeling_qwen3_next.py:743-747`
  ```python
  self.num_experts   = config.num_experts            # E = 512
  self.intermediate_dim = config.moe_intermediate_size  # d_moe = 512
  self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2*self.intermediate_dim, self.hidden_dim))
  self.down_proj    = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
  ```
- **값 충돌**: `E` = `d_moe` = 512. 두 이름이 서로 자리를 바꿔도 숫자로는 안 잡힘.
- **CSV 자리와 오류 (op_id 1114·1115·1119·1120)** — Codex 는 "weight 첫 축"만 지적했으나
  실제로는 **양방향 교환**이다:

  | op | 현재 | 올바름 |
  |---|---|---|
  | 1114/1115 `gate_up_proj` weight | `[d_moe, 2*d_moe, d_model]` | `[E, 2*d_moe, d_model]` |
  | 1119/1120 `down_proj` weight | `[d_moe, d_model, E]` | `[E, d_model, d_moe]` |
  | 1115/1120 group-size 피연산자 | `[d_moe]` | `[E]` |

  즉 첫 축은 `d_moe`→`E`, `down_proj` 마지막 축은 `E`→`d_moe` 로 **정반대**로 붙어 있다.
- **교차 모델 불일치 (결정적)**: 같은 `aten._grouped_mm` 을 쓰는 MiniMax-M2 는 group-size
  피연산자를 `[E]` 로 **정확히** 렌더링한다(op_id 175). 같은 op 이 모델에 따라 갈린다.

## A-6 · Qwen3-Next · chunk mask 축 `d_rope` → `d_chunk` — **CONFIRMED**

- **CSV 자리**: 같은 파일 op_id 125(`ones`), 126(`triu`), 142(`masked_fill`), 963(`eye`),
  982/983 — 전부 `[[d_chunk, d_rope]]`.
- **소스**: `modeling_qwen3_next.py:413,423,432` (`torch_chunk_gated_delta_rule`)
  ```python
  mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, ...), diagonal=0)
  attn = attn + torch.eye(chunk_size, ...)
  ```
  두 축 모두 `chunk_size` → `[d_chunk, d_chunk]`.
- **값 충돌**: `d_rope` = 64 = `d_chunk` = 64.
- **자체 모순**: op 142 에서 이 마스크가 `[B, n_h_lin_v, 1, d_chunk, d_chunk]` 에
  브로드캐스트되므로 두 번째 축은 `d_chunk` 여야만 한다. RoPE 는 이 함수에 등장조차 안 함.

## A-7 · Qwen3-Next · decode `transpose` 입력 `d_head_lin_k` → `d_head_lin_v` — **CONFIRMED**

- **CSV 자리**: `.../full/decode.csv` op_id 131, 330 (전 linear_attn 레이어):
  `input [[B, n_h_lin_v, 1, d_head_lin_k]]` → `output [[B, 1, n_h_lin_v, d_head_lin_v]]`
- **불변식 위반**: `aten.transpose` 는 축을 **재배열만** 한다 — 이름을 바꿀 수 없다.
  마지막 축은 transpose(1,2) 에 영향받지 않는데 입출력 이름이 다르다. 둘 중 하나가 틀렸다.
- **어느 쪽이 틀렸나**: 이 텐서는 recurrent delta-rule 의 `core_attn_out` 으로 value 계보다.
  같은 파일 op_id 100 이 value 텐서를 `[B,1,n_h_lin_v,d_head_lin_v]` →
  `[B,n_h_lin_v,1,d_head_lin_v]` 로 **정확히** 렌더링한다. 131 은 그것을 되돌리는
  transpose 이므로 **입력 쪽 `d_head_lin_k` 가 틀렸다.**
- **값 충돌**: `d_head_lin_k` = `d_head_lin_v` = 128.

## A-9 · NX-AI/xLSTM-7b — **주장 없음** (Codex "이상 없음")

---

# 배치 A 집계

| 판정 | 건수 |
|---|---|
| CONFIRMED | 8 / 8 |
| REJECTED | 0 |
| UNCERTAIN | 0 |

**8건 전부 확정.** 지난 라운드(요약 카드의 "유도 상수" 표를 근거로 준 지적 3건이 전부
오탐)와 정반대 결과다. 원인은 PREAMBLE 수정 — 이번엔 Codex 에게 요약 카드가 아니라
**실제 CSV 행(`5. 대표 트레이스 표본`)**을 줬다.

**8건 중 5건이 "산출물 내부에서 이미 모순"**이라는 점이 중요하다 (A-2 weight↔output,
A-4 q_proj↔split, A-5 모델 간 같은 op 불일치, A-6 브로드캐스트 상대축, A-7 transpose
불변식). 이건 외부 소스 없이 **트레이스 내부만으로 자동 검출 가능한 오류 계열**이라는
뜻이고, 게이트 규칙으로 승격할 후보다.

**공통 원인**: 8건 전부 값 충돌(collision)이다 — 두 심볼이 같은 정수라서 숫자 검증으로는
영원히 안 잡히는 자리. `moe-routed-slot-collision` / `heuristic-fabricated-labels` 계열.

---

# 반영 가능성 점검 — `rules/label_overrides.yaml` 로 8건을 다 적을 수 있나

**결론: 8건 전부 적을 수 있다. 스키마 확장 불필요.**

> **정정 기록.** 처음에 나는 "4건은 표현 불가, 스키마 확장이 필요하다"고 적었다. 틀렸다.
> YAML 머리말 주석이 `model/module/from/to/expect/source` 를 "필수 항목", `layer_types` 를
> 유일한 "선택" 항목으로 적어놓았고, 나는 그 주석을 스키마 전체로 읽었다. 실제
> `src/label_overrides.py` 의 매처는 훨씬 많은 자리 한정자를 지원하며 466개 엔트리 중
> **448개가 이미 그것을 쓰고 있다.** 코드를 먼저 읽었어야 했다.

## 실제로 지원되는 자리 한정자 (`src/label_overrides.py` apply 매처)

| 키 | 의미 |
|---|---|
| `op_type` | 그 op 에만 |
| `nth` | 모듈 안에서 그 op 의 서수 |
| `field` | `i`(input) / `o`(output) / `w`(weight) — **방향을 가른다** |
| `shape_index` | 여러 피연산자 중 몇 번째 |
| `axis` | 축 위치. **음수는 오른쪽부터** |
| `rank` | shape 길이 |
| `shape` | 렌더된 shape 전체가 정확히 일치할 때만 |
| `spread: class` | 모듈 경계를 넘어 **텐서 등가류 전체**를 같은 이름으로 (weight 제외) |

이 조합이면 앞서 "불가"라고 적었던 4건이 전부 해결된다:

- **A-2** → `field: w` + `axis: 1` (그리고 전치된 피연산자는 `field: i`/`axis: 2`).
  `mlp.experts` 안의 정당한 `d_model` 은 축 위치가 달라 건드려지지 않는다.
- **A-4** → `op_type: split` + `field: i` + `shape` 로 그 자리만. k/v 되접기 `view` 는
  shape 이 달라 제외된다.
- **A-5** → 방향별로 엔트리를 나눠 쓴다(`axis: 0` 은 `d_moe`→`E`, `down_proj` 의 마지막
  축은 `E`→`d_moe`).
- **A-7** → `field: i` 가 입력 축만 고른다. 출력 축은 안 건드린다.

## 스코프만으로 되는 것 (4건 — 한정자 없이 모듈 스코프로 충분)

| 건 | 스코프 | 근거 (실측) |
|---|---|---|
| A-1 | `^model\.layers\.\d+\.self_attn$`, `n_h+2*n_kv`→`d_head/2`, expect 64 | 이 이름은 `slice`(248)/`neg`(124)/`concat`(124) = `rotate_half` 자리에만 등장. 총 496건 전부 오류. 정당한 용례 0. |
| A-3 | `^model\.layers\.\d+\.linear_attn\.in_proj_ba$`, `d_chunk`→`2*n_h_lin_v`, expect 64 | 자체 서브모듈이라 스코프가 깨끗함. |
| A-6 | `linear_attn`, `d_rope`→`d_chunk`, expect 64 | linear_attn 안의 `d_rope` 는 `ones`(72)/`triu`(72)/`masked_fill`(36)/`eye`(36)/`elementwise_add`(36) = 청크 마스크 자리에만 등장, 252건 전부 오류. linear attention 에 RoPE 는 없다. 밖의 정당한 `d_rope`(self_attn 264, rotary_emb 15)는 스코프로 보호됨. |
| A-8 | `k_proj`/`v_proj`/`k_norm`, `n_h*d_head`→`n_kv*d_head`, expect 4096 | 셋 다 독립 서브모듈이고 그 안의 해당 이름은 전부 오류. `q_proj`/`q_norm` 은 스코프 밖이라 안전. |

## 자리 한정자가 필요한 것 (4건)

| 건 | 왜 한정자가 필요한가 |
|---|---|
| A-2 | `mlp.experts` 안에서 `d_model` 은 `index`(124)·`masked_fill_`(124)·`transpose`(124)·`grouped_matmul`(124)·`elementwise_mul`(62)·`view`(62)·`sum`(62)·`_to_copy`(62) 에 걸쳐 등장하고 **대부분 정당하다**(down_proj 폭 `[E,d_model,d_moe]`, 출력 `[k*T,d_model]`). 틀린 건 `gate_up_proj` weight 의 **가운데 축 하나**뿐. 일괄 치환하면 나머지를 전부 파괴한다. |
| A-4 | `n_kv*d_head` 는 bare `self_attn` 의 `view`(36)·`split`(12) 에 등장하는데, `view` 중 일부는 k/v 를 `[B,T,n_kv,d_head]` 로 되접는 **정당한** 자리다. 서브모듈로도 op 로도 갈리지 않는다. |
| A-5 | `E` 와 `d_moe` 가 **같은 모듈 안에서 서로 자리를 바꿔** 있다(양방향). 한 방향을 치환하면 다른 방향이 무너진다. |
| A-7 | 오류가 **한 op 의 입력 축에만** 있고 출력 축은 맞다. override 는 방향(입력/출력)을 구분하지 못한다. |


## 결론

**검증도 반영도 병목이 아니다.** 8건 전부 지금 스키마로 반영 가능하다. 다음 단계는
override 8건(실제로는 자리별로 나뉘어 그보다 많은 엔트리)을 쓰고, 재생성 후
`develop/verify_all.py` 가 exit 0 인지, 그리고 각 엔트리가 **발화했는지**(발화 0건이면
게이트 FAIL) 확인하는 것이다.
