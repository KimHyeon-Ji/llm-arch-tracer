# 5개 모델 최종 산출물 검토 요청 — 구조 표현과 라벨이 맞는가

요청일: 2026-09-21. 선행: Kimi-K3 단독 검토 round 1~6 (`codex_answer_kimi_k3_*.md`).

**이번에는 도구나 절차가 아니라 결과물 자체를, 그리고 K3 하나가 아니라 5개 모델 전부를
봐 달라.** 각 모델의 `prefill.csv` / `decode.csv` / `prefill.jsonl` / `decode.jsonl` 이
그 모델의 구조를 제대로 표현하는가, 그리고 안의 라벨이 맞는가.

산출물은 이제 GitHub 에 올라가 있다 (2026-09-21 push):

```
https://github.com/KimHyeon-Ji/llm-arch-tracer/tree/results/models/
```

csv 와 jsonl 은 같은 내용의 두 형식이다. 열:

```
op_id, block_type, repeat, layers, h1..h4(또는 h5), op_type,
input_shape, weight_shape, weight_pos, output_shape,
depends_on, layer_idx, block, sub_block, depth, module_path, raw_op,
params, phase, unmapped, caveat
```

열 개수가 모델마다 다르다(22~25) — 계층 열 `h*` 의 깊이가 모델 구조에 따라 달라서다.

`input_shape` 는 **가중치 피연산자도 포함**하고 `weight_pos` 가 그중 어느 것인지 가리킨다
(FLOPs/바이트를 셀 때 중복으로 세지 않도록).

---

## 0. 검토 범위에서 빼 달라 — 이미 "모른다" 고 적은 것

**미확정은 모델마다 `UNKNOWNS.md` 에 수치로 공개했다.** 그 항목들은 지적해도 새로운
정보가 아니니, **그 밖의 것**만 봐 달라.

| 모델 | 축 자리 | 확정 | scope_inferred | heuristic | open_tie | unresolved | 접힌 질문 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek-V4-Pro | 660,040 | 75.1% | 99,992 | 33,500 | 24,888 | 5,777 | 25 |
| Llama-4-Maverick | 73,550 | 96.7% | 1,728 | — | — | 672 | 2 |
| Kimi-K3 | 5,579,877 | 36.9% | 3,013,269 | 57,419 | 4,584 | 447,947 | 23 |
| gpt-oss-120b | 69,254 | 77.1% | 9,882 | 556 | 5,332 | 72 | 16 |
| gpt-oss-20b | 46,454 | 79.8% | 5,390 | 376 | 3,556 | 48 | 13 |

등급의 뜻:

* `scope_inferred` — scope 정규식만이 후보를 갈랐다. 근거는 있으나 아무도 검증 안 했다.
  구체 크기는 맞다.
* `heuristic` — **산술로 지어낸 이름.** 값은 맞지만 이름이 틀릴 수 있다.
* `open_tie` — 후보 둘 이상이 같은 값이라 트레이스로 못 갈랐다. **이름만** 미정.
* `unresolved` — 이름 붙일 근거가 없어 정수로 뒀다. 주장을 안 하므로 틀릴 것도 없다.

그 밖에 각 `UNKNOWNS.md` 가 공개한 것: ③ 자유 평가 상태(전부 `STALE` 또는 미수행),
판정이 끝난 지적, 표의 숫자가 관측값이 아닌 자리, 그리고 K3 의 경우 배치 전환 대조가
**이 판에 없다**는 사실.

**이 목록에 해당하는 지적은 "이미 공개함" 으로 간주하고 넘어가 달라.
다만 공개 문구 자체가 틀렸다면 그건 지적해 달라.**

---

## 1. 모델별 사실 — 이게 맞는지부터 봐 달라

측정값이다. 내가 잘못 읽은 게 있으면 그것부터 짚어 달라.

### DeepSeek-V4-Pro (`b5968e9190ef611bbf34a7229255be88a0e937c1`)

```
행      prefill 224 / decode 212        열 25 (h1..h5)
접기    MLA+MoE  repeat 2  layers 0-1
        MLA+MoE  repeat 1  layer  2
        MLA+MoE  repeat 29 layers 3,5,7,…,59     (홀수)
        MLA+MoE  repeat 29 layers 4,6,8,…,60     (짝수)
        embed 1 / norm 2 / head 1 / `-` 4행
        -> 2+1+29+29 = 61 = L
심볼    L=61 d_model=7168 n_h=128 n_kv=1 d_head=512 d_ff=3072 V=129280
        E=384 E_shared=1 k=6 d_moe=3072 w_local=128 d_rope=64
        m_csa=4 m_hca=128 n_hc=4 g_o=16 d_g=1024 n_h_I=64 c_I=128 k_I=1024
값 충돌 128: n_h | w_local | m_hca | c_I
        3072: d_ff | d_moe        64: d_rope | n_h_I
        4: m_csa | n_hc           1024: d_g | k_I
```

### Llama-4-Maverick-17B-128E (`10751cb97a4d7c90f7ed89196b98eb8220cfa1c2`)

```
행      prefill 69 / decode 69         열 23
접기    attn+FFN repeat 24 layers 0,2,4,…,46      (짝수 = dense FFN)
        attn+MoE repeat 12 layers 1,5,9,…,45
        attn+MoE repeat 12 layers 3,7,11,…,47
        embed / norm / head 각 1
        -> 24+12+12 = 48 = L
심볼    L=48 d_model=5120 n_h=40 n_kv=8 d_head=128 d_ff=16384 V=202048
        E=128 E_shared=1 k=1 d_moe=8192 chunk_size=8192
값 충돌 128: d_head | E            8192: d_moe | chunk_size
```

### Kimi-K3 (`f831ab66814297da540d832a5235f8e904f29d06`)

```
행      prefill 15,198 / decode 2,334  열 24
접기    attn+FFN repeat 1  layer  0
        (아래 3줄이 8 번 되풀이되며 93 층을 채운다)
          attn+MoE repeat 8 layers 1-2,4-6,8-10  …  49-50,52-54,56-58  (마지막은 5)
          MLA+MoE  repeat 3 layers 3,7,11 … 87,91-92
          attn+MoE repeat 1 layers 12 / 24 / 36 / …
        embed 1 / norm 1 / head 1 / `-` 7행
        -> 층 합계 93 = L, MLA 는 24 개 (마지막 91-92 가 인접)
심볼    L=93 d_model=7168 n_h=96 n_kv=96 d_head=74 d_ff=33792 V=163840
        E=896 E_shared=2 k=16 d_moe=3072 d_moe_lat=3584
        c_kv=512 c_q=1536 d_nope=128 d_v=128 d_rope=64
        n_h_kda=96 d_head_kda=128 d_chunk=64 d_conv=4 n_attn_res_block=12
값 충돌 96: n_h | n_kv | n_h_kda   128: d_nope | d_v | d_head_kda
        64: d_rope | d_chunk
caveat  17,532 행 중 2,944 행이 차 있다 (MoE even-split shim)
```

### gpt-oss-120b (`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`) / gpt-oss-20b (`6cee5e81ee83917806bbde320786a8fb61efebee`)

```
행      각 prefill 51 / decode 51      열 22
접기    120b  attn+MoE repeat 18 layers 0,2,4,…,34   (sliding)
          attn+MoE repeat 18 layers 1,3,5,…,35   (full)     -> 18+18 = 36 = L
       20b  같은 구조로 12+12 = 24 = L
        embed / norm / head 각 1
심볼    d_model=2880 n_h=64 n_kv=8 d_head=64 d_ff=2880 V=201088
        k=4 d_moe=2880 w_local=128 n_sink=1
        E=128 (120b) / E=32 (20b)
값 충돌 2880: d_model | d_ff | d_moe    64: n_h | d_head    128: E | w_local (120b)
```

---

## 2. 봐 줬으면 하는 것

### Q1. 구조 표현이 맞는가 — 접기가 층 구성을 잃지 않는가

`block_type` / `repeat` / `layers` 열이 층 스택을 접는다. 층 번호는 `layers` 열이 들고
있고, 모든 모델에서 층 합계가 `L` 과 정확히 맞는다(위 표에 계산을 적었다).

* 이 접기가 실제 층 구성을 잃지 않고 표현하는가? **`layers` 열의 층 번호가 각 모델의
  config 가 정하는 스케줄과 실제로 맞는지** 소스로 확인해 달라. 특히:
  * **V4-Pro** — 3,5,7,… / 4,6,8,… 두 무리로 갈린 것이 `heavily_compressed_attention` /
    `compressed_sparse_attention` 의 교대와 맞는가? 앞의 0-1, 2 가 따로 떨어진 이유는?
  * **Llama-4** — 짝수 층이 dense FFN, 홀수 층이 MoE 인 것이 맞는가? 그리고 홀수가
    1,5,9,… 와 3,7,11,… 로 다시 갈린 것이 `chunked_attention` 3 + `full_attention` 1
    주기와 맞는가?
  * **gpt-oss** — 짝수/홀수가 sliding/full 교대와 맞는가? (두 무리 다 `attn+MoE` 라고만
    적혀 있어 **이름만 봐서는 어느 쪽이 sliding 인지 알 수 없다** — `layers` 와
    `structure.yaml` 의 `layer_sched` 를 맞춰 봐야 한다. 이게 받는 쪽에 불친절한가?)
  * **K3** — MLA 가 3,7,11,…,87,91-92 의 24 개이고 **마지막 두 개(91,92)가 인접**한
    것이 맞는가? (`full_attn_layers` 가 1-based `[4,8,…,88,92,93]` 이라는 해석이다.)
* **`block_type` 이름이 오해를 부르는가.** K3 는 KDA 층을 `attn`, MLA 층을 `MLA` 로
  쓴다 — 같은 `self_attn` 모듈인데 이름이 갈려 있다. 반대로 V4-Pro 는 서로 다른
  두 attention 이 **둘 다** `MLA+MoE` 다.

### Q2. 값이 겹치는 자리의 라벨 — **오라벨이 숨을 수 있는 유일한 자리다**

위 표의 "값 충돌" 이 각 모델의 위험 지점이다. 그 값이 붙은 축들이 **자리마다 맞는
이름**을 받았는지 봐 달라. 특히:

* **V4-Pro 의 `128`** — `n_h`(head 수) / `w_local`(sliding window) / `m_hca`(HCA 압축
  주기) / `c_I`(indexer 폭) 넷이 같은 값이다. 네 개가 서로 바뀐 자리가 있는가?
* **gpt-oss 의 `2880`** — `d_model` / `d_ff` / `d_moe` 가 전부 같다. expert FFN 폭과
  residual 폭이 바뀐 자리가 있는가?
* **gpt-oss 의 `64`** — `n_h`(=64) 와 `d_head`(=64) 가 같다. head 축과 채널 축이 바뀐
  자리가 있는가? (이 둘이 바뀌면 FLOPs 는 같은데 구조 해석이 완전히 달라진다.)
* **Llama-4 의 `128`** — `d_head` 와 `E`(전문가 수)가 같다.
* **K3 의 세 묶음** — round 3~4 에서 확인해 준 자리(MLA 의 `q_pass`=d_nope,
  `value_states`=d_v, `g_proj` 폭=n_h*d_v, KDA 의 `d_head_kda`)는 반영했다.
  **그 밖의 자리**만 봐 달라.

### Q3. MoE 를 세 가지 방식으로 표현하고 있다 — 일관성이 맞는가

같은 아키텍처 개념(토큰을 전문가로 라우팅)이 모델마다 다르게 나온다:

```
gpt-oss / V4-Pro   grouped_matmul
                   in  [[B*k*T, d_model], [E, d_model, 2*d_moe], [E]]
                   out [[B*k*T, 2*d_moe]]
                   -> 라우팅된 (토큰, 슬롯) 쌍을 B*k*T 로 편다

Llama-4            batched_matmul
                   in  [[E, B*T, d_model], [E, d_model, 2*d_moe]]
                   out [[E, B*T, 2*d_moe]]
                   -> 모든 전문가가 모든 토큰을 계산한다 (E*T)

Kimi-K3            전문가별 matmul (shim)
                   in  [[3840, d_moe_lat], [d_moe_lat, d_moe]]
                   out [[3840, d_moe]]      caveat 열이 차 있다
                   -> 896 중 4 명만 추적, 토큰 균등 분할. 3840 은 B*k*T/4 라는
                      shim 산술이지 아키텍처 폭이 아니다 (이미 공개)
```

* 셋 다 **그 구현이 실제로 하는 일**을 맞게 그린 것인가?
* Llama-4 의 `E*B*T` 는 그 구현이 정말 모든 전문가를 모든 토큰에 대해 도는 것인가,
  아니면 내가 추적 경로를 잘못 골라 그렇게 보이는 것인가?
* `2*d_moe` (gate+up 융합) 표기가 세 모델에서 일관되게 맞는가?

### Q4. 구조가 **빠진** 것은 없는가

표에 있어야 하는데 없는 것:

* **V4-Pro**: CSA/HCA 압축기, Lightning Indexer, mHC(`attn_hc`), grouped o_proj 가
  전부 나타나는가? MLA 의 KV 압축 경로는?
* **Llama-4**: chunked attention 의 청크 경계가 보이는가? QK-norm, attention temperature
  tuning 은?
* **gpt-oss**: attention sink(`n_sink=1`), sliding window, MXFP4 양자화 관련 연산은?
  router 의 `softmax` 입력이 `[B*T, k]` 인데(top-k **뒤**), 이게 맞는 순서인가?
* **K3**: KDA 의 conv1d / forget gate / 청크 스캔, MoE 라우터와 공유 전문가 구별은?
* 모든 모델: `decode` 가 `prefill` 의 축소판으로 일관적인가 (T -> 1, 캐시 길이 T+1)?

### Q5. `weight_pos` 규약이 오해를 부르는가

`input_shape` 두 번째가 곧 `weight_shape` 의 전치인 행이 많다. 예:

```
in  [[B*T, d_model], [d_model, c_q]]
w   [c_q, d_model]     (weight_pos 1)
out [[B*T, c_q]]
```

FLOPs/바이트를 세려는 사람이 이 표기로 **틀리게 셀 여지**가 있는가? 열 이름이나
문서를 어떻게 바꾸면 나은가?

---

## 3. 원하는 답의 형태

* **틀린 라벨**: 어느 모델 / 어느 파일 / 어느 행(`op_id`) / 현재 이름 -> 맞는 이름 /
  소스 근거(파일·줄)
* **구조 표현 문제**: 무엇이 어떻게 오해를 부르는가
* **빠진 것**: 무엇이 없는가
* 확인했는데 맞으면 **맞다고** 적어 달라 — 그것도 기록으로 남긴다

**모델별로 따로 판정해 달라.** 지난 검토에서 "계열에 같은 문제" 라는 지적 6건 중
실제로 맞은 것은 1건이었다. 한 모델에서 본 것을 다른 모델에 옮기지 말아 달라.

산출물은 읽기만 하고 **수정하지 말아 달라.**

## 4. 참고 — 이미 통과한 검사

다시 확인할 필요는 없지만, 무엇이 덮여 있는지 알면 범위를 좁히는 데 도움이 될 것이다.

```
독립 배치 검증            발행 라벨을 다른 배치 크기의 실제 shape 과 대조 -- 어긋난 라벨 0
module-field membership   0 (고정 revision 의 remote source 기준)
reshape 자체 유도 대조     불일치 0
한 축에 이름이 둘 이상인 등가류   0 (하드 불변식, 5개 모델 전부)
n_h 와 n_kv 가 한 shape 에 공존   0 (하드 불변식, 5개 모델 전부)
```

## 5. 아직 **열려 있는** 게이트 항목 — 숨기지 않는다

"전부 통과" 가 아니다. 지금 열려 있는 것을 그대로 적는다. 이것들은 이미 알고 있으니
다시 지적할 필요는 없지만, **검토 결과를 읽을 때 이 상태를 감안해 달라.**

| 모델 | 열린 항목 |
|---|---|
| DeepSeek-V4-Pro | 더 이상 맞지 않는 확인 기록 27건 / 발화하지 않은 라벨 교정 3건 / heur 427→549 |
| Llama-4-Maverick | 발화하지 않은 라벨 교정 4건 |
| Kimi-K3 | 근거 없는 확인 기록 736건(인용 **품질**) / bare 446,563→451,876 (지어낸 이름을 거둬 정수로 물러난 **의도한** 결과) |
| gpt-oss-120b | 더 이상 맞지 않는 확인 기록 6건 |
| gpt-oss-20b | 더 이상 맞지 않는 확인 기록 6건 |

**이 항목들을 조사한 결과 — 그리고 이 진단이 맞는지도 봐 달라.**

"발화하지 않은 라벨 교정" / "맞지 않는 확인 기록" 은 **앵커가 낡은 것**이지 발행된
라벨이 틀렸다는 뜻이 아니다. K3 에서 같은 증상 12건을 조사했더니 라벨 자리는 그대로
있고 op_type 별 서수만 +2 밀려 있었다(내가 Q/K 정규화 식을 커널과 맞추면서 `sum`·`add`
가 새로 생긴 탓). 나머지 네 모델도 방금 조사했다:

* **gpt-oss 120b/20b 의 6건씩, V4-Pro 의 27건 중 다수** — 확인 기록이 기대하는 shape 이
  `['n_h', 'T', 'T']` 처럼 **선행축이 맨 `n_h`** 인데, 지금 표는 `['B*n_h', 'T', 'T']` 로
  접혀 있다. 실제로 세어 보면 선행축이 맨 `n_h` 인 자리는 **0건**, `B*n_h` 는
  gpt-oss-120b 24건 / V4-Pro 48건이다. 즉 B=1 로 잡던 시절에 쓴 앵커가
  배치 접기(B=3 발행) 뒤 안 맞게 된 것이다.
* **Llama-4 의 4건** — 전부 `B -> 1` 교정이다(크기 1 축을 `B` 로 그리지 말라는 것).
  지금은 안 걸린다.

**Q0. 이 진단이 맞는가.** 위 자리들에서 **발행된 라벨 자체는 맞는지** 확인해 달라.
내 진단이 틀렸다면 — 즉 라벨이 실제로 틀렸다면 — 그게 가장 중요한 지적이다.

또한 ③ 자유 평가(규칙이 못 잡는 종류의 오류를 사람/LLM 이 직접 보는 층)는 5개 모델
모두 `STALE` 이거나 미수행이다. **그래서 이 검토를 부탁하는 것이다.**

**추적 범위의 한계**: FakeTensor/CPU reference 경로로 잡았다. 실제 GPU Triton kernel 의
op 구성은 이 산출물의 범위가 아니다. K3 의 KDA 는 `fla` 의 torch reference,
MoE 는 even-split 대체 경로다.
