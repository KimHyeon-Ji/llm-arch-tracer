# R3 — 산출물 **전수** 검토와 정식 publish 승인

요청일 2026-09-27. `develop/PLAN_plus_at_layer.md` v3 의 R3.
R1/R2 를 먼저 돌린 뒤 이 문서를 보냅니다. **표본이 아니라 전수를 봐 주십시오.**

이 라운드가 통과하면 `status: provisional` -> `released` 로 올립니다. 통과 전에는
산출물이 provisional 입니다.

---

## 1. 무엇이 만들어졌나

```
models/moonshotai__Kimi-K3/plus_at/
  prefill.csv   decode.csv   prefill.jsonl   decode.jsonl     derived view
  expected_footprint.jsonl    독립 matcher 가 만든 변경 허용 목록  [입력]
  actual_footprint.jsonl      적용 결과                          [출력]
  symbols.yaml                심볼 범례 (kind 로 architecture / trace_control 구분)
  MANIFEST.json               입력·overlay·도구·출력·source 해시 + V9 승계 표
```

원본 `models/moonshotai__Kimi-K3/{prefill,decode}.{csv,jsonl}` 은 **건드리지 않았습니다.**

```
status               provisional
base_results_commit  86353fd1896524647f793b18c85a9b1f705401de
footprint digest     4bbb87266eb9c9b0d6f6432ae259a50c33a217b01b22d910606817d068939a9a
바뀐 셀              6,864
```

`expected_footprint.jsonl` 과 `actual_footprint.jsonl` 의 sha256 이 **같습니다**
(`4bbb87266eb9c9b0…`) -- 독립 matcher 와 적용기가 같은 셀 집합을 냈다는 뜻입니다.

## 2. 셀 분포 (전수)

sub_id / phase / field 별:

```
k3-kda-nchunk   prefill  input_shape   1040    output_shape  1040
k3-moe-regular  prefill  input_shape    966    output_shape   759
                decode   input_shape    966    output_shape   759
k3-moe-last     prefill  input_shape    322    output_shape   253
                decode   input_shape    322    output_shape   253
k3-moe-concat   prefill  input_shape     92
                decode   input_shape     92
                                        합계  6,864
```

module / op 분포 (상위):

```
2080  k3-kda-nchunk   model.layers.N.self_attn                            exp
 460  k3-moe-regular  model.layers.N.block_sparse_moe.experts.0.act_fn    elementwise_mul
 460  k3-moe-regular  ...experts.1.act_fn                                 elementwise_mul
 460  k3-moe-regular  ...experts.2.act_fn                                 elementwise_mul
 460  k3-moe-last     ...experts.3.act_fn                                 elementwise_mul
 184  k3-moe-concat   model.layers.N.block_sparse_moe                     concat
```

**regular : last = 3450 : 1150 = 3 : 1** 입니다(전문가 0,1,2 대 3). 개수 자체가 규칙의
대응을 확인합니다.

## 3. 검증 결과

```
V1 격리        비대상 canonical cell 전부 original == derived           통과
               (파일 바이트 비교가 아니라 cell 비교. 다만 CRLF 로 되쓰면 원본과
                바이트 동일함을 다섯 모델 전부에서 따로 확인했습니다)
V2 집합 동일성 actual_cell_set == expected_cell_set, digest 일치          통과
V3 sanity      한 shape 에 같은 맨정수가 둘 이상 -> **전체 실패**        위반 0
V4 형태 보존   cell 키 집합 동일 (행·피연산자 수·축 개수 불변)           통과
V5 canonical   csv 와 jsonl 을 같은 파서로 배열화해 비교                통과
V6 점 대입     n_chunk=5 -> 5,  floor(15360/4) -> 3840,
               15360 - 3*3840 -> 3840                                   통과
V7 잔여 보고   아래 4 절                                                 보고됨
V8 의미 맥락   바뀐 셀의 (module, op_type) 이 선언 집합 안              통과
V9 게이트      12 종 재실행 / 3 종 not_applicable / 1 종 승계            통과
멱등·결정론    재적용 시 출력 해시 동일                                  확인
fixture        사람이 쓴 사례 14 + 산술 사례 4, 6/6 시험 통과             통과
```

## 4. V7 — 바꾸지 않고 남긴 맨정수 (숨기지 않습니다)

```
prefill   0:1   2:80  3:80  4:80  5:80  6:80  7:80  8:79  9:72     632 자리
decode    0:1   2:80  3:80  4:80  5:80  6:80  7:80  8:79  9:72     632 자리
                                                          합계  1,264 자리
```

전부 attention-residual 누적 경로입니다(계열 C). **B 만 스윕하면 값이 변하지 않으므로**
1차 범위에서 뺐습니다. 식은 규명됐지만 셀별 semantic stage 는 확정하지 않았습니다.

## 5. V9 게이트 승계 표 — **이 판단을 검토해 주십시오**

`decision` 을 제가 초안으로 적었습니다. deny-by-default 로 두었고 분류되지 않은 게이트가
있으면 release 를 실패시킵니다. `review_status` 는 전부 `proposed` 입니다.

```
rerun (12)
  schema_shape_rank_token_type        라벨 교체가 축 개수·토큰 종류를 바꾸면 안 된다
  symbol_declared                    표에 넣은 이름이 symbols.yaml 에 선언돼 있다
  expression_no_cycle                symbols 의 expr 가 순환 참조하지 않는다
  substitution_nonneg_integer        대입 결과가 정수이며 음수가 아니다
  zero_axis_only_initial_residual    `0` 은 선언된 초기 residual 두 자리에서만
  batch_seq_head_axis_consistency    배치·시퀀스·head 축 위치 일관성
  head_scope_exclusive               n_h / n_kv / n_h_kda 가 한 shape 에 공존하지 않는다
  moe_quotient_remainder_consistency concat 입력의 regular/last 대응이 전문가 순서와 맞는다
  prefill_decode_structure           두 phase 의 공통 구조가 같은 방식으로 바뀌었다
  layers_repeat_consistency          layers 를 펼친 수 == repeat
  row_metadata_preserved             caveat·unmapped·params·depends_on 보존
  op_id_dag                          op_id 유일성, dependency 존재, DAG 비순환

not_applicable (3)  -- **이 셋이 검토 대상입니다**
  axis_class_consistency  등가류는 원시 원장 + 포트 사이드카 위에 선다.
                          발행본 표만으로 재구성할 수 없다
  reshape_derivation      구체 shape 사이드카가 필요하다. 발행본에 없다
  port_coverage           포트 사이드카를 본다. derived view 범위 밖

inherited_unchanged (1)
  rules_fingerprint       라벨 규칙을 바꾸지 않았다. 원본 PASS 를 승계한다
```

## 6. 물어보는 것

### Q1. `not_applicable` 셋이 타당합니까

셋 다 "발행본 표만으로는 재구성 불가" 라는 사유입니다. 이 사유가 맞습니까? 맞다면 그
게이트들이 못 보는 실패가 +@ 층에 남는 셈인데, 그걸 다른 검사로 메워야 합니까?

### Q2. footprint 전수를 보고 이상한 자리가 있습니까

`actual_footprint.jsonl` 6,864 줄이 그대로 있습니다. 각 줄이
`{phase, op_id, field, shape_index, axis, before, after, module_path, op_type, sub_id}`
입니다. 바뀌면 안 될 자리가 섞였는지 봐 주십시오.

특히 확인 부탁: `k3-moe-regular` 와 `k3-moe-last` 의 경계입니다. `experts.3` 만 last 여야
하는데 footprint 의 module_path 로 확인 가능합니다.

### Q3. 표가 자기모순이 아닙니까

`plus_at/prefill.csv` 를 열어 다음을 확인해 주십시오.

```
같은 축이 단독과 곱에서 같은 이름인가
  이전:  단독 `5`  vs  곱 `B*n_h_kda*n_chunk`      <- 모순이었다
  이후:  단독 `n_chunk`  vs  곱 `B*n_h_kda*n_chunk`
MoE concat 의 입력 합이 출력과 맞는가
  입력 3*n_trace_regular + n_trace_last  vs  출력 `B*k*T`
  대입하면 3*3840 + 3840 = 15360 = 3*320*16 -- 맞습니다. 표기가 이 관계를 드러냅니까?
```

### Q4. `symbols.yaml` 의 kind 구분이 읽는 쪽에 전달됩니까

`C_trace` / `n_trace_regular` / `n_trace_last` 를 `kind: trace_control`,
`origin: adaptation` 으로 표시했습니다. 표를 받은 사람이 이걸 아키텍처 심볼과 혼동하지
않겠습니까? caveat 은 모든 해당 행에 그대로 남아 있습니다.

### Q5. 정식 publish 를 승인하십니까

승인하시면 `status: released` 로 올리고 overlay 의 `status: proposed` 를 `accepted` 로
바꿉니다. 조건이 남았다면 무엇인지 적어 주십시오.

## 7. 읽을 곳

```
models/moonshotai__Kimi-K3/plus_at/MANIFEST.json        해시 전부 + V9 표
models/moonshotai__Kimi-K3/plus_at/actual_footprint.jsonl   바뀐 셀 6,864 전수
models/moonshotai__Kimi-K3/plus_at/prefill.csv          적용본
models/moonshotai__Kimi-K3/prefill.csv                  원본 (대조용)
develop/plus_at/overlay-moonshotai__Kimi-K3.yaml        판정 원문
develop/fixtures/plus_at/cases.yaml                     사람이 쓴 fixture
develop/plus_at_apply.py  plus_at_refmatch.py  plus_at_canon.py
develop/test_plus_at.py
```

**읽기만 하고 수정하지 말아 주십시오.**

## 8. 제가 아는 약점

정직하게 적습니다 -- 이걸 공격해 주시면 좋겠습니다.

```
1  refmatch 를 제가 썼습니다. "다른 구현" 이지만 같은 사람의 같은 이해입니다.
   fixture 를 손으로 써서 보완했지만 fixture 도 제가 썼습니다.
   -> 의미 독립성은 R1 에만 걸려 있습니다.
2  sweep_verified 가 거짓입니다. T=384 재트레이스를 하지 않았습니다.
   식이 이 한 점에서만 확인됐습니다.
3  선언 집합이 틀린 경우 V8 은 통과합니다(선언을 그대로 믿습니다).
   mutation test 와 negative control 로 일부 메웠지만 구멍이 남습니다.
4  계열 C 1,264 자리는 손대지 않았습니다. 사용자가 d_chunk 나 층 배치를 스윕하면
   그 자리는 틀립니다.
```
