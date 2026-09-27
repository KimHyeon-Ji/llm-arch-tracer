# R2 — 이 판정들을 **깨뜨려 주십시오**

요청일 2026-09-27. `develop/PLAN_plus_at_layer.md` v3 의 R2.
**R1 을 먼저 보내고 답을 받은 뒤에 이 문서를 보냅니다.** R1 에 답하신 세션에는 보내지 마십시오.

---

## 이 라운드의 규칙

R1 과 달리 이번에는 **제 판정을 전부 공개합니다.** 부탁은 하나입니다.

> **이 판정들이 틀렸을 경우를 찾아 주십시오.** 맞다는 확인은 필요 없습니다.

## 판정 넷

### A. `5` -> `n_chunk` (prefill 2,080 자리)

```
선택자   module `\.self_attn$`  /  op_type `exp`
         shape 전체 일치 ["B","n_h_kda","5","d_chunk","d_head_kda"]
         field input_shape·output_shape  /  shape_index 0  /  axis 2
식       n_chunk = T / d_chunk = 320 / 64 = 5
유효조건 T % d_chunk == 0
분류     architecture
```

근거 셋:

```
1  fla/ops/kda/naive.py:108-109 -- naive_chunk_kda 가 T 를 NT = T//BT 개 청크로 쪼갠다
                                   (BT = chunk_size, 기본 64)
2  src/build_table.py:1250      -- 트레이서가 이미 T == d_chunk*n_chunk 를 등재하고 있다
                                   (_MERGE_IDENTITY_EQUIVALENTS)
3  발행본 자신                   -- 같은 축을 곱 안에서는 이미 `B*n_h_kda*n_chunk` 로 쓴다
                                   (3,168 자리 / 1,056 행). 단독일 때만 `5` 로 남았다.
                                   즉 표가 자기모순이었다.
```

### B-1. `3840`/`12` -> `n_trace_regular` (4,784 자리 중 3,450)

```
선택자   module `\.block_sparse_moe\.experts\.(0|1|2)(\.|$)`  /  axis 0
식       n_trace_regular = floor(N_route / C_trace)
         N_route = B*T*k (prefill) = 15360,  B*k (decode) = 48
         C_trace = trace.expert_cap = 4
유효조건 N_route >= C_trace,  C_trace == 4
분류     trace_control  (아키텍처 심볼이 아니다)
```

### B-2. `3840`/`12` -> `n_trace_last` (1,150 자리)

```
선택자   module `\.block_sparse_moe\.experts\.3(\.|$)`  /  axis 0
식       n_trace_last = N_route - (C_trace - 1) * floor(N_route / C_trace)
근거     src/kda_shim.py:641-647 -- per = max(1, n // traced) 이고
         end = n if i == traced-1 else min(n, start+per)
         즉 **마지막 전문가가 나머지를 받는다.** 이 트레이스는 15360/4 가 딱 나눠져
         regular 와 last 가 같은 값이지만 식은 다르다.
```

### B-3. 상위 concat 의 피연산자 대응 (184 자리)

```
선택자   module `\.block_sparse_moe$`  /  op_type concat  /  field input_shape  /  axis 0
규칙     피연산자 0..C_trace-2 -> regular,  마지막 피연산자 -> last
근거     src/kda_shim.py:649 -- outs = cat(outputs, dim=0),
         outputs 는 for i in range(traced) 순서로 쌓인다
```

### 바꾸지 않은 것

```
계열 C  residual 누적 폭 2..9 와 빈 residual 0    1,264 자리
        B 만 스윕하면 값이 안 변하므로 1차 범위 밖.
        식은 규명됐다: R_before(l) = ceil(l/S), R_after(l) = floor(l/S)+1,
        S = attn_res_block_size = 12, 최종 = ceil(L/S)+1 = 9 (L=93).
        발행본 1,264 자리 전부 이 식들의 합집합으로 설명되고 불일치 0.
        **다만 여러 식이 같은 값을 낼 수 있어 셀별 semantic stage 는 확정하지 않았다.**
```

## 반증 조건 (제가 적어 둔 것)

```
A     T=384, d_chunk=64 로 재트레이스하면 이 축의 구체값이 6 이어야 한다. 5 면 틀렸다.
B-1   C_trace 를 5 로 바꿔 트레이스하면 이 축이 floor(15360/5) = 3072 여야 한다.
B-2   N_route % C_trace != 0 인 설정(예: k=15)에서 regular 와 last 의 값이 달라야 한다.
B-3   concat 출력 축 0 이 N_route 와 같아야 한다. 다르면 피연산자 대응이 틀렸다.
```

**이 반증 조건들이 충분합니까?** 더 쉽게 깨뜨릴 방법이 있으면 그것을 알려 주십시오.

## negative control (제가 확인한 것)

```
A 가 건드리면 안 되는 자리
  발행본 residual 계열의 `5`  (module 이 root, op 이 concat/sum/softmax/…)   160 자리
  self_attn 안이지만 op 이 exp 가 아닌 `5`
  원시 원장의 [B, n_h_kda, 5, d_chunk, 5] 의 축 4 (causal prefix)
  -- 이 shape 은 **발행본에는 없다**(확인: n_h_kda + 맨정수 2개 이상인 자리 0 건)

B 가 건드리면 안 되는 자리
  [3840, d_moe] 의 축 1 (d_moe -- 이미 심볼)
  experts.3 을 regular 로, experts.0..2 를 last 로 잘못 가르는 경우
```

## 유효 범위의 한계 (제가 인정하는 것)

```
point_verified        참. 이 트레이스의 값(B=3, T=320, k=16, d_chunk=64, C_trace=4)에서
                      식에 대입하면 원본 구체값이 정확히 복원된다.
sweep_verified        **거짓.** T·k·d_chunk·C_trace 를 바꿔 다시 트레이스하지 않았다.
semantic_evidence     R1/R2 가 끝나기 전에는 거짓으로 둔다.
```

즉 **틀린 식도 한 점에서는 우연히 맞을 수 있다**는 것을 알고 있습니다. 그 점을 공격해
주십시오.

## 물어보는 것

### Q1. 어느 판정이 가장 약합니까

넷 중 가장 먼저 깨질 것을 지목하고, 왜 그런지 적어 주십시오.

### Q2. 제 선택자가 잡지 말아야 할 자리를 잡습니까

`module_regex` 와 `shape` 전체 일치로 골랐습니다. 이 조합이 놓치는 반례가 있습니까 --
특히 **제가 선언한 (module, op_type) 집합 자체가 틀린** 경우입니다. 그 경우 제 검사(V8)는
선언을 그대로 믿으므로 통과합니다.

### Q3. `C_trace` 를 표에 넣은 것이 옳습니까

`n_trace_regular` / `n_trace_last` 는 `C_trace`(= 트레이스가 돌린 전문가 수)에 의존합니다.
`C_trace` 를 `kind: trace_control` 로 분리 표시했지만, 그래도 표의 축 이름에 트레이스
산물이 들어왔습니다. 이게 받아들일 만합니까, 아니면 리터럴을 두고 사이드카에만 적어야
합니까?

### Q4. 계열 C 를 안 바꾼 것이 옳습니까

식은 규명됐는데 셀별 stage 는 확정하지 않았습니다. "규명된 식을 표에 반영하지 않고 두는
것" 이 더 안전합니까, 아니면 미검증 상태로라도 사이드카에 적어야 합니까?

### Q5. 놓친 것

위 다섯 갈래 밖에서 제가 못 본 실패 방식이 있으면 지적해 주십시오.

## 읽을 곳

```
develop/plus_at/overlay-moonshotai__Kimi-K3.yaml     판정 원문 (근거·exemplar·검증상태)
models/moonshotai__Kimi-K3/plus_at/                  적용본과 footprint
models/moonshotai__Kimi-K3/prefill.csv               원본
src/kda_shim.py:633-652
fla/ops/kda/naive.py:100-170
```

**읽기만 하고 수정하지 말아 주십시오.**
