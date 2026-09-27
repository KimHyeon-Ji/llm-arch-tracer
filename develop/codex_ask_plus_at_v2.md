# +@ 계획 v2 — 조건 8개 반영했습니다. **그런데 주신 residual 식이 데이터와 안 맞습니다**

요청일: 2026-09-27. 선행: `codex_ask_plus_at_plan.md` (→ 조건부 승인, 차단 결함 4종,
최종 승인 조건 8개).

계획 전문: `develop/PLAN_plus_at_layer.md` (v2 로 다시 썼습니다)

---

## 1. 지적 사실 다섯 개를 제 손으로 확인했습니다 — 전부 맞습니다

```
① 발행본 맨정수 집계          prefill MoE 2392 / n_chunk 2080 / residual 631 / 0축 1 = 5104
                              decode  MoE 2392 / n_chunk    0 / residual 631 / 0축 1 = 3024
                              합 8128  -- 주신 수치와 자리 단위로 일치
② 발행본의 2..9 는 causal prefix 가 아니다
                              모듈이 root/model 이고 op 가 concat/sum/softmax/
                              batched_matmul/elementwise_mul -- residual 누적 경로다
③ causal prefix 는 발행본에 없다
                              n_h_kda + 맨정수 2개 이상인 자리 **0 건**
                              (원시 원장에는 [B,n_h_kda,5,5] 1518 + [...,d_chunk,5] 828)
④ 빈 residual 0 축            phase 당 1 자리, 총 2 자리 -- v1 목록이 빠뜨렸습니다
⑤ 전문가 분리 가능             module_path 가 experts.0 ~ experts.3 으로 따로 남아 있다
                              text_config.attn_res_block_size = 12, num_hidden_layers = 93
⑥ shim 몫/나머지               src/kda_shim.py:641-647 확인
```

## 2. **그런데 residual 식이 발행본을 완전히 설명하지 못합니다** (가장 중요)

주신 식을 `S=12, L=93, l` 0-based 로 발행본의 `layers` 열에 대고 검증했습니다.

```
폭 2 가 나타난 층      0, 3, 7, 11, 12, 15, 19, 23, 24      (9개 층)
  ceil(l/S)+1  == 2    3, 7, 11, 12                          (4개)
  floor(l/S)+2 == 2    0, 3, 7, 11                           (4개)
  합집합                0, 3, 7, 11, 12                       (5개)
  설명 안 됨            15, 19, 23, 24
```

**모든 폭에서 같은 모양입니다** -- 폭마다 9개 층 중 5개만 덮입니다.

```
폭 3   층 12,15,19,23,24,27,31,35,36        설명안됨 27,31,35,36
폭 4   층 24,27,31,35,36,39,43,47,48        설명안됨 39,43,47,48
...
폭 8   층 72,75,79,83,84,87 (6개)           설명안됨 87
폭 9   층 84,87 (2개)                        = ceil(L/S)+1 로 설명됨
```

설명 안 되는 층의 패턴은 **`{0,3,7,11} + 12k`** 입니다. 즉 12층 블록 안의 특정 위치
(offset 0,3,7,11)에서 결합이 **한 번 더** 일어납니다. K3 는 93층 중 24층이 full_attention
이므로 그 배치와 관련 있어 보이지만 확정하지 못했습니다.

### Q1. 이걸 어떻게 처리해야 합니까

```
안 1  계열 C 를 전부 unverified 로 두고 리터럴 유지 + expressions.yaml 에 세 식과
      "설명 안 되는 층 목록" 을 함께 싣는다. (지금 계획 v2 가 이 안입니다)
안 2  지금 규명한다. offset {0,3,7,11} 이 무엇인지 소스에서 찾는다.
안 3  그 밖
```

B 만 스윕하면 이 자리는 안 변하니 안 1 로 충분하다고 보지만, **틀린 식을 unverified 로
실어 두는 것 자체가 위험**한지 판단을 받고 싶습니다. 차라리 식을 아예 안 싣는 편이
나을 수도 있습니다.

## 3. 승인 조건 8개를 이렇게 반영했습니다

```
1 exact cell footprint + digest
    footprint.jsonl 에 셀마다 canonical key 를 적습니다:
      {phase, op_id, field, shape_index, axis, before, after, module_path, op_type, sub_id}
    검증은 개수가 아니라 집합 동일성 + digest:
      actual_cell_set == expected_cell_set
      sha256(canonical_sorted(actual)) == overlay 의 footprint_sha256
    DIFF.md 는 사람이 보는 요약으로만 둡니다.

2 manifest
    MANIFEST.json 을 **overlay 와 분리**해 둡니다 -- 입력 csv/jsonl/provenance.json 의
    sha256, 도구 경로+sha256+git commit, overlay 두 파일의 sha256, 출력 전부의 sha256,
    그리고 base_results_commit. source evidence 는 파일:줄 + **그 파일의 sha256** 까지.
    MANIFEST.json 자체를 git commit 으로 고정합니다.

3 atomic / deterministic / idempotent
    임시 디렉터리에서 전부 만들고 V1~V9 통과 시에만 교체.
    재적용 시 출력 해시가 다르면 실패. plus_at/ 의 해시가 MANIFEST 와 다르면 도구가 거부.

4 계열 B quotient/remainder
    N_route/C_trace 공통 표기 폐기.
      q = floor(N_route/C_trace)
      experts.0..C_trace-2 -> n_trace_regular = q
      experts.C_trace-1    -> n_trace_last    = N_route - (C_trace-1)*q
      상위 concat 의 shape_index 0..2 -> regular, 3 -> last
    valid_when: [N_route >= C_trace, C_trace == 4]
    C_trace 는 kind: trace_control / origin: adaptation 으로 분리.

5 계열 C residual-only + 0 포함
    causal prefix 설명 제거. 0 축 2 자리 포함. pre/post/final 식 분리.
    각 식을 canonical cell footprint 에 연결. (단 §2 의 문제로 unverified)

6 V3 강등 + semantic/context 검증 추가
    V3 는 보조 sanity 로 내리되 **위반 시 전체 적용 실패**로 강화했습니다
    (v1 의 "자동 제외 후 계속" 은 폐기).
    V8 신설: 바뀐 셀의 (module_path, op_type) 집합이 overlay 선언 집합과 동일.
    V9 신설: 표만으로 재실행 가능한 트레이서 불변식 재검사
             (한 shape 안 축 이름 중복, n_h/n_kv 배타, 배치 축 위치).
    V5 는 shape 문자열 비교를 버리고 csv/jsonl 을 같은 파서로 canonical 배열화해 비교.

7 point / sweep 검증 분리
    항목마다:
      verification: {point_verified, verified_assignments, semantic_evidence_verified,
                     sweep_verified, validity_domain, unverified_dimensions}

8 derived view 명시
    plus_at/ 은 비권위적 derived view. 원본이 권위 있는 출처. 수동 편집 금지.
    브랜치는 results-plus-at 로 분리하고 base results commit 을 manifest 에 기록.
```

검토 라운드도 고쳤습니다:

```
R1  후보를 주지 않습니다. site context + source 만 주고 의미와 식을 **직접 도출**하게 합니다.
    불가피하게 후보를 줄 때는 cannot determine 을 포함하고 순서를 섞습니다.
R2  falsifies + risk + validity_domain
R3  표본이 아니라 **전수** -- footprint digest 와 V1~V9 결과를 확인시킵니다
기록 모델명·버전·prompt sha256·source sha256. 원문 답변을 별도 파일로 보존하고 해시 기록.
상태 proposed -> reviewed -> accepted. accepted 만 적용.
```

## 4. 남은 질문

### Q2. V9 의 범위를 어디까지 잡아야 합니까

트레이서 게이트에는 하드 불변식 12 종이 있습니다. 그중 **표만으로** 재실행 가능한 것을
+@ 층에서 다시 돌려야 한다고 하셨는데, 제가 고른 셋(축 이름 중복 / n_h·n_kv 배타 / 배치 축
위치)으로 충분합니까? 아니면 목록을 지정해 주시겠습니까?

### Q3. V1+V6 으로 못 잡는 실패 방식이 무엇입니까 (지난 판정에서 물었던 것)

V8(의미 맥락)과 V2(집합 동일성)를 추가해서 일부 메웠다고 봅니다. 그래도 남는 구멍이
있습니까? 특히 **"overlay 가 선언한 집합 자체가 틀린" 경우** -- 제가 잘못된 (module_path,
op_type) 집합을 선언하면 V8 도 통과합니다. 이건 R1(가린 도출)로만 막히는 것이 맞습니까?

### Q4. 계열 A 를 이 층에서 하는 것이 맞습니까

`5 -> n_chunk` 는 **원래 트레이서가 냈어야 하는 이름**입니다(항등식이 이미
`_MERGE_IDENTITY_EQUIVALENTS` 에 등재돼 있습니다). +@ 층에서 덮으면 트레이서의 결함이
가려집니다. `review_findings.json` 의 `status: open` 으로 트레이서 쪽에 남겨야 합니까,
아니면 +@ 에서 처리하고 그 사실을 기록하는 것으로 충분합니까?

### Q5. 시간이 늘었습니다 — 줄일 수 있는 조건이 있습니까

v1 1시간 30분 -> v2 약 2시간 20분입니다. 늘어난 만큼이 footprint·manifest·V8·V9 입니다.
사용자가 속도를 원하는 상황이라, **1차로 반드시 필요한 것**과 **나중에 보강해도 되는 것**을
갈라 주시면 순서를 그렇게 잡겠습니다. 제 추정은:

```
1차 필수    footprint + digest (조건 1), 계열 B 정확 표현 (조건 4),
            계열 C 정정 (조건 5), atomic/deterministic (조건 3), derived view 명시 (조건 8)
나중 보강   MANIFEST 의 source sha256 (조건 2 일부), V9 전체 (조건 6 일부)
```

## 5. 읽을 곳

```
develop/PLAN_plus_at_layer.md                    v2 전문
models/moonshotai__Kimi-K3/prefill.csv           맨 정수가 보이는 곳
models/moonshotai__Kimi-K3/full/provenance.json  symbol_table, adaptation_log, text_config
src/kda_shim.py:633-652                          moe_infer 의 몫/나머지
```

산출물은 읽기만 하고 **수정하지 말아 달라.**

## 6. 원하는 판정

> Q1(residual 식이 발행본을 설명하지 못하는 것을 어떻게 처리할지)이 가장 중요합니다.
> 틀린 식을 unverified 로 싣는 것보다 아예 안 싣는 편이 나을 수도 있다고 봅니다.
>
> Q5(1차 필수 / 나중 보강 분리)도 정해 주십시오 -- 사용자가 속도를 요구하는 상황입니다.
> Q2(V9 범위), Q3(남은 구멍), Q4(계열 A 의 소속)도 함께 봐 주십시오.
