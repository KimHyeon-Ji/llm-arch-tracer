# 배치 C 판정 — Codex CSV/JSONL 라벨 주장 검증

대상: Qwen__Qwen3.6-27B, Qwen__Qwen3.6-35B-A3B, bzantium__tiny-deepseek-v3
근거 소스: 설치된 transformers 5.14.1. 판정일 2026-09-03.

## 중요 사실

Qwen3.6-27B / 35B-A3B 는 `provenance.json` 상 아키텍처가 `Qwen3_5ForConditionalGeneration`
계열이다 → **modeling 소스가 Qwen3.5 와 같다.** 따라서 C-2~C-7 은 배치 B 와 동일한 결함이
동일한 자리에 있다(`rule-set-converges`).

## 판정

| # | 모델 | 주장 | 판정 | 비고 |
|---|---|---|---|---|
| C-1 | Qwen3.6-27B | rotate_half `n_h+2*n_kv`→`d_rope/2` | **CONFIRMED** | partial rotary 라 `d_head/2` 아님. 행 자체가 자명: 입력 `[B,n_h,T,d_rope]` 를 slice 한 결과가 head-count 식일 수 없다. 256행 |
| C-2 | Qwen3.6-27B | chunk mask `d_rope`→`d_chunk` | **CONFIRMED** | B-5 와 동일 |
| C-3 | Qwen3.6-27B | transpose value 축 | **CONFIRMED, 범위 확대** | Codex 는 decode 만. 전수 검사 결과 **96행, 양쪽 phase** |
| C-4 | Qwen3.6-35B | conv 채널 `2*n_h*d_head`→`2*d_k_lin+d_v_lin` | **CONFIRMED** | B-4 와 동일 |
| C-5 | Qwen3.6-35B | fused-Q `n_kv*d_head`→`2*d_head` | **CONFIRMED** | A-4/B-1 과 동일 |
| C-6 | Qwen3.6-35B | chunk mask | **CONFIRMED** | B-5 와 동일 |
| C-7 | Qwen3.6-35B | transpose value 축 | **CONFIRMED, 범위 확대** | **120행, 양쪽 phase** |
| C-8 | tiny-deepseek-v3 | MLA value 경로 `d_nope`→`d_v` | **REJECTED** | 아래 참조 |
| C-9 | tiny-deepseek-v3 | q/k RoPE 조각 `d_head`→`d_rope` | **REJECTED** | 아래 참조 |
| C-10 | tiny-deepseek-v3 | router/assignment `E`→`k` | **CONFIRMED, 범위 훨씬 넓음** | 아래 참조 |
| C-11 | tiny-deepseek-v3 | `histc` `[E]→[E]` 를 `[k]→[E]` 로 | **CONFIRMED** | C-10 의 일부 |

## C-8 REJECTED — value 경로는 이미 정확하다

`models/bzantium__tiny-deepseek-v3/full/prefill.csv` layer 0 의 value 계보를 op 단위로
따라간 결과 **전 구간이 이미 `d_v`** 다:

    op  66 split_with_sizes [B,n_h,T,d_nope+d_v] -> [B,n_h,T,d_nope], [B,n_h,T,d_v]
    op  94 concat           -> [B,n_h,T,d_v]
    op 119 batched_matmul   [n_h,T,T] x [n_h,T,d_v] -> [n_h,T,d_v]
    op 124 view             [B,T,n_h,d_v] -> [B,T,n_h*d_v]
    op 127 matmul (o_proj)  [T,n_h*d_v] x [n_h*d_v,d_model] -> [T,d_model]

소스(`modeling_deepseek_v3.py:434` `k_pass, value_states = torch.split(k_pass,
[self.qk_nope_head_dim, self.v_head_dim], dim=-1)`)와 정확히 일치한다. `d_nope` 가 나오는
자리(op 46/91/92/93/95/96/98/104~110)는 전부 **query/key 경로의 `d_nope+d_rope`** 로 맞다.
고칠 것이 없다.

## C-9 REJECTED — `d_head` 라벨이 이 모델에 존재하지 않는다

prefill 전체를 토큰 단위로 스캔했으나 `d_head` 는 **0건**이다. RoPE 조각과 rotary table 은
이미 `d_rope` 로 렌더링돼 있다.

## C-10/C-11 CONFIRMED — 다만 Codex 범위보다 훨씬 넓다

이 모델은 **E = k = 8**(모든 전문가로 라우팅)이라 두 이름이 값으로 구분되지 않는다.
prefill 은 `k*T` = 8·17 = 136 ≠ 8 이라 라우팅 슬롯 축이 구분되어 **정확히 렌더링**된다:

    prefill  view [T,k]->[k*T] / sort [k*T] / histc [k*T]->[E] / grouped_matmul [k*T,d_model]

decode 는 T=1 이라 `k*T` = 8 = `E` 로 붕괴하고, 라우팅 영역 **전체**가 `E` 로 잘못 붙었다:

    decode   view [B,E]->[E] / sort [E] / histc [E]->[E] / grouped_matmul [E,d_model]
             div_ [B,k],[B,1] -> [B,E]      <- 나눗셈이 k 를 E 로 바꿀 수 없다
             elementwise_mul [B,E]->[B,E]   <- [B,k] 여야 한다
             index [k,d_model],[k] -> [E,d_model]

Codex 는 "router 및 expert assignment 축"과 `histc` 만 짚었으나, 실제로는 `mlp.gate` 의
`div_`/`elementwise_mul` 부터 `mlp.experts` 의 sort/floor_divide/index/histc/cumsum/ge/
unsqueeze/clamp_/masked_fill_/grouped_matmul/split/empty_like/arange/index_put_/view 까지
**decode 라우팅 영역 전체**다.

추가로 **prefill 에도 한 자리 남아 있다**: 마지막 `view [k*T,d_model] -> [T,E,d_model]` 의
가운데 축은 토큰당 선택된 전문가 수이므로 `k` 다(E 와 값이 같아 가려졌다).

이건 `moe-routed-slot-collision` 메모리가 적어둔 바로 그 계열이고, prefill/decode 대조로
푼다(`src/tdep.py`).

---

**집계: CONFIRMED 9 / REJECTED 2 / 11건.**

Qwen3.6 두 모델은 override 10개 작성 → 재트레이스 → promote 완료
(27B 발화 5,184 · 35B 발화 7,380, 미발화 0, 전치 위반 0, C-FAIL 0).
tiny-deepseek-v3 의 C-10/C-11 은 아직 미작성.
