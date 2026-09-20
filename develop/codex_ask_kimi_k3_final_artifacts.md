# Kimi-K3 최종 산출물 검토 요청

요청일: 2026-09-20. 선행: round 1~4 (`codex_answer_kimi_k3_*.md`).
고정 revision `f831ab66814297da540d832a5235f8e904f29d06`.

**검토 대상은 최종 산출물 네 파일이다.** 이번에는 도구나 절차가 아니라 **결과물 자체**를
봐 달라 — 이 표가 Kimi-K3 의 구조를 제대로 표현하는가, 그리고 안의 라벨이 맞는가.

```
../llm-arch-tracer-results/models/moonshotai__Kimi-K3/
  prefill.csv    15,198 행   4.4 MB
  decode.csv      2,318 행   1.4 MB
  prefill.jsonl  15,198 행  10.0 MB
  decode.jsonl    2,318 행   2.7 MB
  structure.yaml            심볼 표
  UNKNOWNS.md               미확정 공개
```

csv/jsonl 은 같은 내용의 두 형식이다. 열:

```
op_id, block_type, repeat, layers, h1..h4, op_type,
input_shape, weight_shape, weight_pos, output_shape,
depends_on, layer_idx, block, sub_block, depth, module_path, raw_op,
params, phase, unmapped, caveat
```

`input_shape` 는 가중치 피연산자도 포함하고 `weight_pos` 가 어느 것인지 가리킨다
(FLOPs/바이트를 셀 때 중복으로 세지 않도록).

---

## 0. 검토 범위에서 빼 달라 — 이미 "모른다" 고 적은 것

**미확정은 `UNKNOWNS.md` 에 수치로 공개했다.** 그 항목들은 지적해도 새로운 정보가 아니니,
**그 밖의 것**만 봐 달라. 공개한 내용:

| 등급 | 자리 | 공개 문구 |
|---|---:|---|
| `scope_inferred` | 3,010,923 | scope 정규식만이 후보를 갈랐다. 구체 크기는 맞다 |
| `heuristic` | 57,419 | **산술로 지어낸 이름.** 값은 맞지만 이름이 틀릴 수 있다 |
| `open_tie` | 4,584 | 후보 둘 이상이 같은 값이라 트레이스로 못 갈랐다. **이름만** 미정 |
| `unresolved` | 447,947 | 이름 붙일 근거가 없어 정수로 뒀다 |

축 자리 5,572,977 개 중 확정 2,052,104 개(36.8%). 그 밖에 공개한 것:

* 접힌 질문 **23 개** (`UNKNOWNS.md` 1 절) — 후보가 안 갈린 자리들
* `근거 없는 확인 기록 736 건` — `label_confirmed.yaml` 의 인용 **품질** 문제
* `bare 퇴행 446,563 -> 451,876` — 지어낸 이름을 거둬 정수로 물러난 의도한 결과
* MoE even-split shim 으로 **대체된 숫자** — `caveat` 열이 찬 행이 prefill 1,472 /
  decode 1,472. 896 전문가 중
  4 명만 추적하고 토큰을 균등 분할한다. 전문가당 토큰 수 3,840 은 `B*k*T/4` 라는 shim
  산술이지 아키텍처 폭이 아니다
* 배치 전환(B=1 → B=3) 대조의 한계 5 가지 (7 절) — 수치 시험이지 대수적 증명이 아님 등

**이 목록에 해당하는 지적은 "이미 공개함" 으로 간주하고 넘어가 달라.** 다만 **공개 문구
자체가 틀렸다면** 그건 지적해 달라.

---

## 1. 봐 줬으면 하는 것

### Q1. 구조 표현이 맞는가

`block_type` / `repeat` / `layers` 열이 93 층 스택을 이렇게 접는다:

```
attn+MoE   repeat 1 / 5 / 8   (행 13,590)
attn+FFN   repeat 1           (행    846)
MLA+MoE    repeat 3           (행    752)
embed / norm / head           (각 1)
```

K3 는 `linear_attn_config.full_attn_layers`(1-based)로 층을 가른다. 0-based 로 MLA 는
**3, 7, 11, …, 87, 91, 92 의 24 개**, 나머지 69 개가 KDA 다.

* 이 접기가 실제 층 구성을 잃지 않고 표현하는가?
* `attn+MoE` 와 `MLA+MoE` 가 같은 MoE 를 쓰는데 이름이 갈려 있다. KDA 층을 `attn`,
  MLA 층을 `MLA` 로 쓴 것인데, 받는 쪽이 오해할 표기인가?
* **마지막 두 MLA 층이 붙어 있다.** `full_attn_layers` 가 1-based 로
  `[4, 8, …, 88, 92, 93]` 이라 0-based 로는 `…, 87, 91, 92` 다 — 91 과 92 가 연속이고
  4 간격이 거기서 깨진다. 이게 의도된 구성인가(마지막 층을 full attention 으로 두는 관례),
  아니면 내가 스케줄을 잘못 편 것인가? 이 해석이 `block_type` 접기와 층 유형별 라벨
  스코프의 전제다.

### Q2. 값이 겹치는 자리의 라벨

이 모델은 세 묶음이 값으로 구별되지 않는다. **오라벨이 숨을 수 있는 유일한 자리다.**

```
 64 : d_rope,  d_chunk
 96 : n_h,     n_kv,     n_h_kda
128 : d_nope,  d_v,      d_head_kda
```

round 3~4 에서 당신이 확인해 준 것(MLA 의 `q_pass`=d_nope / `value_states`=d_v,
`g_proj` 폭=n_h*d_v, KDA 의 `d_head_kda`)은 반영했다. **그 밖의 자리**에서 이 세 묶음이
서로 바뀐 곳이 있는지 봐 달라. 특히:

* KDA 층과 MLA 층이 같은 모듈 이름(`self_attn`)을 쓴다 — 한쪽 이름이 다른 쪽에 샜는가?
* `d_rope`(64) 와 `d_chunk`(64): RoPE 폭과 KDA 청크 크기가 서로 바뀐 자리가 있는가?
* `n_kv`(96): K3 는 MLA 라 KV 가 latent 로 압축되는데, `n_kv` 가 붙은 축이 실제로
  KV head 개수를 뜻하는 자리인가?

### Q3. 대표 행이 옳게 읽히는가

```
MLA (layer 3)  matmul
  in  = [[B*T, d_model], [d_model, c_q]]
  w   = [c_q, d_model]
  out = [[B*T, c_q]]

KDA (layer 0)  batched_matmul
  in  = [[B*n_h_kda*n_chunk, d_chunk, d_head_kda],
         [B*n_h_kda*n_chunk, d_head_kda, 1]]
  out = [[B*n_h_kda*n_chunk, d_chunk, 1]]
```

* MLA 쪽: `c_q`(=1536) 가 `q_a_proj` 출력이 맞는가? `weight_pos` 규약대로 `input_shape`
  두 번째가 곧 `weight_shape` 의 전치인데, 이 표기가 혼동을 주는가?
* KDA 쪽: `B*n_h_kda*n_chunk` (=3·96·5=1440) 가 접힌 배치 축으로 맞는가? round 2 에서
  당신이 확인해 준 형태인데, 최종 산출물에서도 같은지 봐 달라.

### Q4. 구조가 **빠진** 것은 없는가

표에 있어야 하는데 없는 것이 있는가. 예를 들어:

* MLA 의 KV 압축 경로(`kv_a_proj_with_mqa` → `kv_b_proj`)가 온전히 나타나는가?
* KDA 의 conv1d, forget gate, 청크 스캔이 전부 보이는가?
* MoE 라우터(`KimiMoEGate`)와 공유 전문가가 구별되어 나타나는가?
* `decode` 가 `prefill` 의 축소판으로 일관적인가 (T=320 → 1, 캐시 길이 T+1)?

### Q5. `structure.yaml` 심볼 표

```
L=93  d_model=7168  n_h=96  n_kv=96  d_head=74  d_ff=33792  V=163840
ctx=1048576  E=896  E_shared=2  k=16  d_moe=3072  d_moe_lat=3584
c_kv=512  c_q=1536  d_nope=128  d_v=128  d_rope=64
n_h_kda=96  d_head_kda=128
(null: d_shared, n_grp, w_local, chunk_size, n_sink, layer_sched, m_csa, m_hca, g_o, d_g)
```

* config 와 대조해 틀린 값이 있는가?
* `d_head=74` 가 눈에 띈다 — MLA 의 `d_nope+d_rope`=192 도 `d_v`=128 도 아니다. 이 심볼이
  이 모델에서 무엇을 가리켜야 하는가, 아니면 null 이어야 하는가?
* `null` 로 둔 것 중 실제로는 값이 있어야 하는 것이 있는가?

---

## 2. 원하는 답의 형태

* **틀린 라벨**: 어느 파일 / 어느 행(op_id) / 현재 이름 → 맞는 이름 / 소스 근거(파일·줄)
* **구조 표현 문제**: 무엇이 어떻게 오해를 부르는가
* **빠진 것**: 무엇이 없는가
* 확인했는데 맞으면 **맞다고** 적어 달라 — 그것도 기록으로 남긴다

산출물은 읽기만 하고 **수정하지 말아 달라.**

## 3. 참고 — 이미 통과한 검사

다시 확인할 필요는 없지만, 무엇이 이미 덮여 있는지 알면 범위를 좁히는 데 도움이 될 것이다.

```
게이트 C1~C17            FAIL 0
독립 배치 검증            발행 B=3 라벨을 B=4 실제 shape 과 대조 -- 어긋난 라벨 0
module-field membership   0 (고정 revision 의 remote source 기준)
reshape 자체 유도 대조     불일치 0
배치 전환 B=1 -> B=3      373,155 레코드 재실행 대조, 불일치 0
                          (근거: full/batch_transition_proof.json)
```

## 4. 재현

```powershell
cd ..\llm-arch-tracer-results\models\moonshotai__Kimi-K3
# prefill.csv / decode.csv / prefill.jsonl / decode.jsonl / structure.yaml / UNKNOWNS.md
```

트레이스 범위: FakeTensor/CPU reference KDA (`fla` 의 torch reference) 와 MoE even-split
대체 경로. 실제 GPU Triton kernel 의 op 구성은 이 산출물의 범위가 아니다.
