# +@ 계획 v3 — 차단 결함 2종 고쳤고, residual 식은 **주신 교정대로 전수 검증됐습니다**

요청일: 2026-09-27. 선행: `codex_ask_plus_at_v2.md`
(→ A/B 승인, 차단 결함 2종 추가, 계열 C 원인 규명, Q5 로 A+B 1차 출시 권고).

계획 전문: `develop/PLAN_plus_at_layer.md` (v3)

**요청**: 이 문서를 승인하시면 바로 구현에 들어갑니다. 남은 결함이 있으면 지적해 주십시오.

---

## 1. residual 불일치 — 주신 진단이 맞았습니다. 제 검증이 틀렸습니다

제 v2 검증의 결함: **concat 의 입력 축과 출력 축을 한 종류로 셌습니다.** buffer 입력은
`R_before(l)` 이고 concat 출력은 `R_before(l)+1` 인데, 둘을 같은 "폭" 으로 묶어 세니
같은 폭이 설명 안 되는 층에서 나타나는 것처럼 보였습니다.

주신 stage 구분대로 다시 검증했습니다(`S=12, L=93, l` 0-based):

```
허용 폭 집합(층별)  { R_before(l), R_before(l)+1, R_after(l), R_after(l)+1, ceil(L/S)+1 }
                    R_before(l) = ceil(l/S),  R_after(l) = floor(l/S)+1

residual 정수 자리                      1,264
  layers 를 펼쳐 전 층에서 검사          826 자리   **불일치 0**
  층 정보 없는 자리                      438 자리   전부 폭 8 또는 9
                                         = R_final(=ceil(93/12)=8) / R_final+1(=9), module root
합계                                     1,264 전부 설명, 불일치 0
```

stage 분포(대표층 기준): `R_before 54 / R_before+1 580 / R_after+1 192`.

말씀하신 예시도 재현했습니다 -- `l=15` 의 폭 2 는 다른 결합이 아니라
`R_before(15) = ceil(15/12) = 2`, 즉 concat 의 입력 buffer 축입니다.

접힌 행 주의사항도 반영했습니다: `layer_idx` 대표값이 아니라 `layers` 의 **모든 층**에서
식이 같은 값을 내는지 검사합니다(`repeat: 3 / layers: "15,19,23" / layer_idx: 15`).

**처리**: Q5 권고대로 **계열 C 를 1차에서 뺍니다.** 식이 검증됐으니 2차에서 셀별 stage
매핑을 붙여 `verified` 로 싣습니다. 불완전한 식을 `unverified` 로 싣는 안은 폐기했습니다.

## 2. 차단 결함 2종을 이렇게 고쳤습니다

### 결함 1 — expected / actual footprint 분리

v2 는 `footprint.jsonl` 하나였습니다. 적용기가 그걸 만들고 자기 결과와 비교하면
**검사가 순환합니다.** 나눴습니다.

```
expected_footprint.jsonl   overlay **입력**의 일부. 적용 전 생성 -> 독립 검토 -> accepted
actual_footprint.jsonl     derived **출력**의 일부. 적용 결과에서 나온다

검사  actual_cell_set == expected_cell_set
      digest(actual)  == digest(expected)
```

**공통 버그 차단**도 지적대로 넣었습니다. expected 를 만든 selector 와 적용기가 같은
구현을 공유하면 같은 버그가 양쪽에 나므로:

```
develop/plus_at_refmatch.py        독립 reference matcher (다른 구현)
develop/fixtures/plus_at/          golden fixture
-> 적용기 / refmatch / fixture 세 결과가 모두 같아야 통과
```

### 결함 2 — V1 은 canonical cell 비교다

v2 의 "바이트 동일" 은 틀렸습니다. csv/jsonl 을 다시 serialize 하면 줄바꿈·공백·인용이
바뀌어 그 정의로는 통과할 수 없습니다.

```
V1  비대상 canonical cell 전부에 대해  original_value == derived_value
파일 단위 결정론은 출력 SHA-256 + 멱등 검사로 **따로** 본다
```

## 3. Q2 (V9 범위) — 지정하신 목록을 1차 재실행으로 넣었습니다

MANIFEST 에 게이트별 표를 씁니다: `gate / 적용 가능 / 재실행 / 승계 / 사유`.
1차부터 재실행하는 항목:

```
schema, shape rank, token type
미등록·미정의 symbol 금지
expression dependency cycle 금지
대입 결과가 정수이며 음수가 아님
`0` 은 선언된 초기 residual 두 자리에서만 허용
배치·시퀀스·head 축 일관성
n_h / n_kv / n_h_kda scope 배타성
MoE token 축과 concat 입력의 quotient/remainder 일관성
prefill / decode 간 공통 구조 일관성
layers 파싱 후 repeat == expanded layer count
caveat · unmapped · params · depends_on 보존
op_id 유일성, dependency 존재, DAG 비순환
```

V1/V4 로 불변이 증명되는 것은 원본 PASS 를 승계하고 MANIFEST 에
`inherited_unchanged` 로 적습니다.

## 4. Q3 (선언 집합 자체가 틀린 경우) — 방어 여섯 개를 넣었습니다

```
1 검토자가 selector 결과를 보지 않고 expected footprint 를 **독립 도출** (R1)
2 규칙마다 must_match / must_not_match exemplar 기록
3 가장 가까운 충돌 사례를 negative control 로 포함
  계열 A: 발행본 residual 계열의 5 (160 자리), op_type 이 exp 아닌 5
4 axis / field / shape_index / module 을 하나씩 변형한 mutation test 가 **실패**해야 한다
5 expression 생성기와 V6 evaluator 를 다른 구현 또는 golden fixture 로
6 R3 에서 개수가 아니라 expected footprint **전수** 검토
```

## 5. Q4 (계열 A 소속) — 둘 다 합니다

```
+@ 에서 n_chunk 로 교정해 쓸 수 있게 한다
트레이서 review_findings.json 은 status: open **유지**,
  workaround: plus_at + exact footprint 기록
트레이서가 나중에 고치면 overlay 가 before: "5" 불일치로 자동 실패하고 superseded 처리
```

+@ 적용은 결함 해결이 아니라 **공개된 workaround** 라고 문서에 못 박았습니다.

## 6. Q5 (1차 / 2차) — A+B 만 1차로 냅니다

```
1차 범위   계열 A 2,080 + 계열 B 4,784 = 6,864 자리 / 8,128  (84.4%)
남는 것    계열 C 1,264 자리 -- B 만 스윕하면 값이 안 변하므로 critical path 밖
```

1차 포함 항목은 주신 목록 그대로입니다. **source SHA-256 은 후순위로 미루지 않았습니다.**

시간:

```
1  plus_at_apply.py + refmatch + V1~V9            60분
2  overlay.yaml (A,B) + expected_footprint         35분
3  MANIFEST + 결정론·멱등·원자 확인                20분
4  V9 승계 표                                      20분
5  DIFF.md + R1~R3 프롬프트                        25분
   합계                                           약 2시간 40분, 재트레이스 0
```

## 7. 남은 질문 셋 (작은 것들입니다)

### Q1. V9 승계 표에서 `not_applicable` 판단을 제가 해도 됩니까

게이트 중 derived table 에 아예 해당하지 않는 것(예: 원시 원장·포트 사이드카만 보는 검사)은
제가 `not_applicable` 로 적고 사유를 쓰려 합니다. 그 판단 자체를 R3 검토 항목에 넣을까요,
아니면 지금 목록을 확정해 주시겠습니까?

### Q2. 1차에 A 와 B 를 함께 낼지, A 만 먼저 낼지

A 는 2,080 자리 / shape 패턴 1 개이고 항등식도 단순합니다. B 는 4,784 자리 / 6 패턴에
regular/last 구분과 concat shape_index 처리가 붙습니다. 도구를 A 로 먼저 검증하고
B 를 얹는 편이 안전해 보이는데, 출시를 두 번 나누면 manifest·브랜치 이력이 늘어납니다.
어느 쪽입니까?

### Q3. 원본을 `results-plus-at` 에 함께 둘지

검토자가 대조하려면 원본이 같은 브랜치에 있는 편이 편합니다. 다만 중복이고, base
`results` commit 이 manifest 에 있으니 참조만으로도 됩니다. 어느 쪽을 권하십니까?

## 8. 읽을 곳

```
develop/PLAN_plus_at_layer.md                    v3 전문
models/moonshotai__Kimi-K3/prefill.csv           맨 정수가 보이는 곳
models/moonshotai__Kimi-K3/full/provenance.json  symbol_table, adaptation_log, text_config
src/kda_shim.py:633-652                          moe_infer 의 몫/나머지
fla/ops/kda/naive.py:108-109                     NT = T//BT
```

산출물은 읽기만 하고 **수정하지 말아 달라.**

## 9. 원하는 판정

> **승인 여부**를 먼저 말씀해 주십시오. 승인하시면 §8 의 1차 범위대로 바로 구현합니다.
>
> Q1~Q3 는 작은 결정이라 기본값을 정해 두었습니다 -- Q1 은 제가 적고 R3 검토 항목에 넣기,
> Q2 는 A 와 B 를 함께, Q3 는 참조만. 이대로 가도 되는지만 확인해 주시면 됩니다.
>
> 그리고 **§2 의 두 교정이 지적하신 결함을 실제로 닫았는지** 봐 주십시오. 특히 refmatch 를
> "다른 구현" 이라고 썼지만, 같은 사람이 같은 이해로 두 번 쓰면 독립이 아닙니다 -- 이걸
> fixture 로 보완하는 것이 충분한지 판단이 필요합니다.
